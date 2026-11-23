#!/bin/bash
#SBATCH --partition=hp_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/xsum-%j.out

cd /data/home/scyb288/run/zy/ELF || exit 1

# 强制 HuggingFace 的 hub / datasets / transformers 不联网，只用本地缓存或下面显式指定的本地路径
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 调用 launch.sh 脚本，用 eval 子命令（评测流程）
# train_xsum_ELF-B.yml 主配置文件（training 配置，但通常同时含模型结构、优化器、数据路径、采样步数等，eval 复用其中的模型与数据字段）
# ELF-B-xsum-torch 加载已训练好的 ELF-B XSum 权重目录
# encoder_model_name 指定条件编码用的 T5 模型路径；ELF 用冻结 T5 encoder 产生 contextual embedding 作为连续空间的“目标表示”
# tokenizer_name 分词器路径，同样用本地 t5-small，保证编解码词表与训练一致
# eval_data_path 评测数据目录；xsum_validation_t5 是已用 T5 分词预处理过的 XSum 验证集
# 全局批大小 1；只评 4 条样本；用 bfloat16 混合精度（数值稳定性优于 fp16）；关掉 torch.compile；不上报 Weights & Biases
NGPU=1 bash scripts/launch.sh eval \
  src/configs/training_configs/train_xsum_ELF-B.yml \
  --checkpoint_path /data/home/scyb288/run/zy/ELF/assets/ELF-B-xsum-torch \
  --config_override encoder_model_name=/data/home/scyb288/run/zy/ELF/assets/t5-small \
  --config_override tokenizer_name=/data/home/scyb288/run/zy/ELF/assets/t5-small \
  --config_override eval_data_path=/data/home/scyb288/run/zy/ELF/assets/xsum_validation_t5 \
  --config_override global_batch_size=1 \
  --config_override num_samples=4 \
  --config_override use_bf16=true \
  --config_override use_compile=false \
  --config_override use_wandb=false