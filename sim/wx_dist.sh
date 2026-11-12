#!/bin/bash
# wx_dist.sh <tag> <rank> <world> <master_internal_ip> [KEY=VAL ...]: one node of a multi-node DAgger run (start the
# same command on every node with rank 0..world-1; rank 0 writes ~/wx/<tag>.pt). Data-parallel: per-node rollouts and
# buffers, all-reduced gradients. Then use wx_run.sh evalc/video on rank 0.
TAG=$1; R=$2; W=$3; M=$4; shift 4
bash $(dirname "$0")/wx_run.sh ${WX_MODE:-dagger} $TAG WX_WORLD=$W WX_RANK=$R WX_MASTER=$M SEED=$((R + 1)) "$@"
