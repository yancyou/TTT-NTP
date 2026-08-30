# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import ast
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
import inspect
from typing import Any, Dict, List

os.environ["MODELING_BACKEND"] = "hf"

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

import hf_models  # noqa: F401

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
try:
    from veomni.data.constants import IGNORE_INDEX
except ImportError:
    from veomni.utils.constants import IGNORE_INDEX
from veomni.data import data_transform as _data_transform
process_pretrain_example = getattr(_data_transform, "process_pretrain_example", None) or getattr(
    _data_transform, "process_plaintext_example", None
)
process_sft_example = getattr(_data_transform, "process_sft_example", None) or getattr(
    _data_transform, "process_conversation_example", None
)
process_pretokenized_example = getattr(_data_transform, "process_pretokenized_example", None)
if process_pretrain_example is None or process_sft_example is None:
    raise ImportError("Installed veomni package does not provide required text data transform functions.")
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
try:
    from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
except ImportError:
    from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _pop_dict_cli_arg(arg_name: str) -> Dict[str, Any] | None:
    if arg_name not in sys.argv:
        return None
    idx = sys.argv.index(arg_name)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"{arg_name} expects a value.")
    raw = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    parsed = None
    for fn in (json.loads, ast.literal_eval):
        try:
            candidate = fn(raw)
            if isinstance(candidate, dict):
                parsed = candidate
                break
        except Exception:
            continue
    if parsed is None:
        raise ValueError(f"{arg_name} expects a dict-like string, got: {raw}")
    return parsed


def _compute_train_steps_compat(args, dataset_length):
    if hasattr(args.train, "compute_train_steps"):
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)
        return args.train.train_steps

    if args.data.datasets_type == "mapping" and dataset_length is not None:
        train_steps = math.floor(dataset_length / args.train.dataloader_batch_size)
    else:
        if getattr(args.train, "dyn_bsz", True):
            train_size = int(args.data.train_size * (1 + args.train.bsz_warmup_ratio / 2))
            train_steps = math.ceil(train_size / (args.train.global_batch_size * args.data.max_seq_len))
        else:
            train_sample = getattr(args.data, "train_sample", 10_000)
            train_steps = math.ceil(train_sample / args.train.dataloader_batch_size)

    if getattr(args.train, "max_steps", None) is not None and train_steps >= args.train.max_steps:
        return args.train.max_steps

    return train_steps


def _build_dataloader_compat(args, train_dataset, train_steps):
    dataloader_kwargs = dict(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        # Compatibility across veomni variants:
        rmpad=getattr(args.train, "rmpad", False),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        dyn_bsz_margin=getattr(args.train, "dyn_bsz_margin", 0),
        dyn_bsz_buffer_size=getattr(args.data, "dyn_bsz_buffer_size", getattr(args.train, "dyn_bsz_buffer_size", 500)),
        dyn_bsz=getattr(args.train, "dyn_bsz", getattr(args.train, "rmpad", True)),
    )
    try:
        from veomni.data.data_loader import DATALOADER_REGISTRY

        builder = DATALOADER_REGISTRY[args.data.dataloader_type]
        dataloader_kwargs = _filter_kwargs_for_callable(builder, dataloader_kwargs)
    except Exception:
        pass
    return build_dataloader(dataloader_type=args.data.dataloader_type, **dataloader_kwargs)


def main():
    foundation_override = _pop_dict_cli_arg("--model.foundation")

    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(Arguments)
    if foundation_override is not None:
        if args.model.foundation is None:
            args.model.foundation = {}
        args.model.foundation.update(foundation_override)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "pretokenized":
        if process_pretokenized_example is None:
            raise NotImplementedError("Installed veomni package does not provide `process_pretokenized_example`.")
        transform = partial(
            process_pretokenized_example,
            input_ids_key=args.data.text_keys,  # text_keys is used as input_ids_key for pretokenized
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    train_steps = _compute_train_steps_compat(args, dataset_length)
    train_dataloader = _build_dataloader_compat(args, train_dataset, train_steps)

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        config_kwargs=args.model.foundation,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        # save model_assets before training
        if args.data.data_type in ["plaintext", "pretokenized"]:
            model_assets = [model_config, tokenizer]
        else:
            model_assets = [model_config, chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter_kwargs = dict(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=getattr(args.train, "rmpad", False),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        gc_steps=getattr(args.train, "gc_steps", 0),
    )
    environ_meter = helper.EnvironMeter(**_filter_kwargs_for_callable(helper.EnvironMeter, environ_meter_kwargs))

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // train_steps
        start_step = global_step % train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {train_steps}, epochs: {args.train.num_train_epochs}"
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            length_in_batch = torch.tensor(0, dtype=torch.int32, device=get_device_type())
            for micro_batch in micro_batches:
                length_in_batch += torch.sum(micro_batch["labels"] != IGNORE_INDEX)
            length_in_batch = all_reduce(length_in_batch, op="sum", group=get_parallel_state().fsdp_group)

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)

                length_in_micro_batch = torch.sum(micro_batch["labels"] != IGNORE_INDEX)
                loss: "torch.Tensor" = (
                    model_outputs.loss * length_in_micro_batch / length_in_batch * get_parallel_state().dp_size
                )

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}", refresh=False
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step, save_async=args.train.save_async)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step, save_async=args.train.save_async)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # Wait for any pending async DCP save to finish flushing to disk before the
    # HF conversion reads it. dcp.async_save writes `.metadata` LAST, so without
    # this the rank-0 HF conversion can FileNotFoundError on a still-writing ckpt.
    # The future is per-rank; all ranks wait, then barrier before the rank-0 read.
    if getattr(Checkpointer, "dcp_save_future", None) is not None:
        logger.info_rank0("Waiting for pending async DCP save to flush before HF conversion...")
        Checkpointer.dcp_save_future.result()
        Checkpointer.dcp_save_future = None
    if dist.is_initialized():
        dist.barrier()
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
