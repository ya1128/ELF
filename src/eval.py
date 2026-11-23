#!/usr/bin/env python
"""Evaluation script for trained ELF models: load a checkpoint and generate text samples."""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__ 是 .../ELF/src/eval.py；外层 dirname 两次找到仓库根 ELF
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # 把仓库根目录塞进 sys.path，使顶层包名能被找到

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder  # 编码器加载入口
from modules.model import ELF_models  # ELF 主模型的注册表
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_checkpoint  # 权重加载
from utils.train_utils import TrainState, get_optimizer
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id  # 数据集加载
from generation import test_generation_uncond, test_generation_cond  # 生成调用入口
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs  # 配置三件套

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model by generating text samples")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--config_override", action="append", default=[],
                        help="Override config values (field_name=value). Repeatable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --seeds is not specified)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to evaluate (e.g. '42,123,456'). Overrides --seed.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint file (e.g. outputs/elf_b-owt/checkpoint_19000) or HF repo id.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    return parser.parse_args()

# 单卡时什么都不做，launch.sh 单卡走的是 exec python ...（不是 torchrun），裸 python 不会设 WORLD_SIZE
def _init_distributed():
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Eval only gathers CPU objects (text strings), so gloo is sufficient.
        dist.init_process_group(backend="gloo")


def main():
    args = parse_args()  # 解析参数
    _init_distributed()  # 多卡分布式初始化

    #  device 与 matmul 精度；允许 fp32 矩阵乘用 TF32 提速（对 fp32 的 T5 编码器有影响）
    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_for_0("Loading configuration...")
    # 配置加载 + override 参数生效
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    # batch size 三分支：优先指定 Global Batch Size；优先指定 Local Batch Size；两个都没指定则直接报错
    world = dist.get_world_size() if dist.is_initialized() else 1  # 开启了分布式训练则返回显卡总数
    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size for evaluation: {config.global_batch_size}")
        total_batch_size = config.global_batch_size  # 总批次
        local_batch_size = total_batch_size // world  # 单卡批次 = 总批次 ÷ 显卡数
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * world  # 总批次 = 单卡批次 × 显卡数
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")
    # 配置体检表：把最终生效值打印出来
    log_for_0(f"Config loaded from {args.config}")
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Max length: {config.max_length}")
    log_for_0(f"Max input length: {config.max_input_length}")
    log_for_0(f"Num samples: {config.num_samples}")
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")
    log_for_0(f"BF16 autocast (sampling): {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"torch.compile (eval model): {bool(getattr(config, 'use_compile', False))}")
    # --seeds 为 None → seed_list = [42]（默认为 seed 值）；此 list 决定第 11 节外层循环跑几轮
    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds is not None else [args.seed]
    log_for_0(f"Seeds to evaluate: {seed_list}")

    log_for_0("Loading tokenizer...")
    # tokenizer_name 被 override，从本地目录读
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    # 把 config.pad_token 这个字符串（'eos' 或 'pad'）翻译成整数 id
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")
    # 数据集加载，先置 None（无条件生成时允许为 None，没有数据）
    eval_dataset = None
    if config.eval_data_path is not None:
        log_for_0("Loading dataset for conditional generation...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(
                config.eval_data_path, tokenizer,
                input_key="input", output_key="output",
            )
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval dataset size: {len(eval_dataset)}")

    # Encoder (HuggingFace T5) 编码器加载
    log_for_0(f"Loading Encoder: {config.encoder_model_name}...") 
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)  # 编码器固定用 fp32 加载
    encoder = encoder.to(device).eval()  # .to(device) 搬到 GPU，.eval() 关 dropout、切推理模式 
    for p in encoder.parameters():
        p.requires_grad_(False)  # 冻结 T5 encoder 提供条件/目标表示（不参与梯度）

    # ELF model （ELF 主模型构建：每个构造参数的来源）
    log_for_0(f"Creating {config.model} model...")
    vocab_size = tokenizer.vocab_size
    # ELF_models[config.model]：ELF_models 是个 dict（注册表），用字符串名取出模型类，再加括号实例化
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)

    # Train state template (only used to plumb EMA params + step/epoch). 预先建好 EMA 参数容器用于接收后续参数
    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_ema(model), step=0, epoch=0,
        dropout_generator=g,
    )
    # 从相应文件读采样超参并覆盖
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    # args.checkpoint_path 为上传的 assets/ELF-B-xsum-torch；
    log_for_0(f"Loading checkpoint from: {args.checkpoint_path}")
    # load_checkpoint  把磁盘权重灌进 state，返回 (state, 第二个值)
    state, _ = load_checkpoint(args.checkpoint_path, state)
    state.model = state.model.to(device).eval()
    # 多卡（分布式）环境，获取当前显卡的编号；如果没开分布式（单卡），就默认当前是 0号卡
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 外层循环：遍历多个随机种子；log_for_0：自定义函数，len(seed_list) > 1 时只有 0 号卡才打印这行日志
    for seed_idx, seed_val in enumerate(seed_list):
        if len(seed_list) > 1:
            log_for_0(f"\n{'#' * 70}")
            log_for_0(f"Seed {seed_idx + 1}/{len(seed_list)}: {seed_val}")
            log_for_0(f"{'#' * 70}")

        # Per-rank offset so ranks generate different samples when sharding;
        # rank 0 keeps the original seed for single-GPU reproducibility.
        # 计算每张卡专属的种子，保了生成的样本互不相同；
        per_rank_seed = seed_val + rank * 1_000_003
        seed_gen = torch.Generator(device="cpu").manual_seed(per_rank_seed) 
        torch.manual_seed(per_rank_seed)  # 同一次运行 batch 顺序、调用次数完全一致，可复现；改 batch size、数据顺序、采样配置循环，这个 t_steps 随机序列可能会变
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(per_rank_seed)
        # 输出目录的管理：先把最初始的输出路径存入变量
        original_output_dir = config.output_dir
        if len(seed_list) > 1:
            config.output_dir = os.path.join(original_output_dir, f"seed_{seed_val}")
        # 内层循环 + 生成调用点
        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
            common_kwargs = dict(
                state=state, tokenizer=tokenizer, generator=seed_gen,
                config=config, sampling_config=sc,
                batch_size=local_batch_size, num_samples=config.num_samples,
            )
            if eval_dataset is None:
                test_generation_uncond(**common_kwargs)
            else:
                # test_generation_cond 负责“取 batch + 用 encoder 编条件 + 拼输入”，再调用内部的采样函数（run_generation）做积分与解码
                test_generation_cond(
                    **common_kwargs, encoder=encoder, dataset=eval_dataset,
                )

        config.output_dir = original_output_dir

    log_for_0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
