#!/usr/bin/env bash
# Launcher for the PyTorch ELF port.
# 整个项目的统一启动入口，负责三件事——解析“跑什么模式 + 用哪个配置”；拼好 Python 入口脚本和分布式参数；用 exec 把进程交给 python 或 torchrun
# Single GPU / CPU:
#     bash scripts/launch.sh train  src/configs/training_configs/train_owt_ELF-B.yml
#     bash scripts/launch.sh eval   src/configs/training_configs/train_owt_ELF-B.yml  embedded-language-flows/ELF-B-owt
#
# Multi-GPU (single-host):
#     NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml

# Multi-host (torchrun rendezvous):
#     NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=node-0 MASTER_PORT=29500 \
#         bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml

# -e：任何命令失败立即终止脚本；-u：引用未定义变量即报错；-o pipefail：管道中任一环节失败，整条管道返回失败
set -euo pipefail
# 至少要有两个参数：模式 + 配置文件；[extra args...] 是可选的透传参数
if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/launch.sh <train|eval> <config.yml> [extra args...]"
    exit 1
fi

MODE=$1  # 第一个参数：train 或 eval
CONFIG=$2  # 第二个参数：yml 配置文件路径
shift 2  # 把前两个参数“弹掉”，剩下的是透传参数
EXTRA="$@"  # 剩余全部参数，拼成一个字符串

# Pick up the script we want to run.
case "$MODE" in
    train) ENTRY=src/train.py ;;
    eval)  ENTRY=src/eval.py ;;
    *) echo "Unknown mode: $MODE (expected 'train' or 'eval')"; exit 1 ;;
esac

NGPU=${NGPU:-1}   # 每节点进程数（GPU 数），默认 1
NNODES=${NNODES:-1}  # 节点数，默认 1
NODE_RANK=${NODE_RANK:-0}  # 当前节点编号，默认 0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}  # 主节点地址，默认本机
MASTER_PORT=${MASTER_PORT:-29500}  # rendezvous 端口，默认 29500

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# 单卡/单进程走裸 python，不经过 torchrun；多卡时才用 torchrun
if [[ "$NGPU" == "1" && "$NNODES" == "1" ]]; then
    echo "[launch] single-process: python $ENTRY --config $CONFIG $EXTRA"
    exec python "$ENTRY" --config "$CONFIG" $EXTRA
else
    echo "[launch] torchrun nproc_per_node=$NGPU nnodes=$NNODES node_rank=$NODE_RANK $ENTRY"
    exec torchrun \
        --nproc_per_node="$NGPU" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$ENTRY" --config "$CONFIG" $EXTRA
fi
