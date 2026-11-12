#!/bin/bash
# wx_run.sh <mode> <tag> [KEY=VAL ...]: one Isaac Lab play job on the wrist task. Log ~/wx/<mode>_<tag>.log
#   dagger : online DAgger of the student from the teacher (BITIDY_BCRNN + BITIDY_RESIDUAL); writes ~/wx/<tag>.pt
#   evala  : (a) teacher, privileged state           evalb : (b) teacher on coarse overhead poses (no images)
#   evalc  : (c) student WX_STUDENT (default ~/wx/<tag>.pt)
#   video  : zoomed render (4 envs) of BITIDY_SCRIPT (default the student) -> ~/wx/vid_<tag>/
# Env: WX_N (envs; 512 dagger / 1024 eval), SEED, WX_CAMS (auto), WX_ENVFILE, any WX_* / BITIDY_* (geometry) override.
export PATH=$HOME/.local/bin:$PATH
# the wx_*.py next to this script (WX_SRC overrides) go into the package
SRC=${WX_SRC:-$(cd "$(dirname "$0")" && pwd)}
export BITIDY_BCRNN=${BITIDY_BCRNN:-$HOME/bcx_e1A_r3.pt} BITIDY_RESIDUAL=${BITIDY_RESIDUAL:-$HOME/cbres2_400.pt}
MODE=$1; TAG=$2; shift 2
for kv in "$@"; do export "$kv"; done
# WX_ENVFILE=<file of KEY=VAL lines> (e.g. the new geometry ~/cb3_geo.env) is exported too; explicit KEY=VAL args win
if [ -n "$WX_ENVFILE" ]; then set -a; . "$WX_ENVFILE"; set +a; for kv in "$@"; do export "$kv"; done; fi
mkdir -p ~/wx
cd ~/IsaacLabTutorial
cp ~/bimanual/*.py $SRC/wx_*.py src/isaaclab_tutorial/tasks/bimanual/
CK=$(find logs -name 'model_*.pt' -path "*bitidy_cup_basket/*dummy*" | sort -V | tail -1)
TASK=BiTidy-CupBasket-Wrist-SO101; CB=install_eval; N=${WX_N:-1024}; SD=${SEED:-5}
case $MODE in
  fsac) CB=install_fsac; N=${WX_N:-512}; SD=${SEED:-1}; export WX_OUT=${WX_OUT:-$HOME/wx/${TAG}_fs.pt} ;;
  dagger) CB=install_dagger; N=${WX_N:-512}; SD=${SEED:-1}; export WX_OUT=${WX_OUT:-$HOME/wx/$TAG.pt} ;;
  evala) export BITIDY_SCRIPT=residual:ResidualScript WX_CAMS=${WX_CAMS:-0} ;;
  evalb) export BITIDY_SCRIPT=wx_student:CoarseTeacher WX_CAMS=${WX_CAMS:-0} ;;
  evale) export BITIDY_SCRIPT=wx_est:EstTeacher WX_CAMS=${WX_CAMS:-1} ;;
  evalc) export BITIDY_SCRIPT=${WX_EVALC_SCRIPT:-wx_student:StudentScript} WX_STUDENT=${WX_STUDENT:-$HOME/wx/$TAG.pt} ;;
  video) export BITIDY_SCRIPT=${BITIDY_SCRIPT:-wx_student:StudentScript} WX_STUDENT=${WX_STUDENT:-$HOME/wx/$TAG.pt} \
           WX_VIDEO_DIR=${WX_VIDEO_DIR:-$HOME/wx/vid_$TAG} BITIDY_RENDER_STEPS=${BITIDY_RENDER_STEPS:-1800} \
           BITIDY_EYE=${BITIDY_EYE:-0.16,-0.55,0.40} BITIDY_LOOK=${BITIDY_LOOK:-0.16,0,0.05}
         TASK=$TASK-Render; CB=install_video; N=${WX_N:-4}; SD=${SEED:-9} ;;
  inputs) export BITIDY_SCRIPT=${BITIDY_SCRIPT:-wx_student:StudentScript} WX_STUDENT=${WX_STUDENT:-$HOME/wx/$TAG.pt}            WX_VIDEO_DIR=${WX_VIDEO_DIR:-$HOME/wx/inp_$TAG} BITIDY_RENDER_STEPS=${BITIDY_RENDER_STEPS:-1350}            BITIDY_EYE=${BITIDY_EYE:-0.17,-0.95,0.80} BITIDY_LOOK=${BITIDY_LOOK:-0.17,0,0.02}
         TASK=$TASK-Render; CB=install_inputs; N=${WX_N:-4}; SD=${SEED:-9} ;;
  segdata) export BITIDY_SCRIPT=${BITIDY_SCRIPT:-residual:ResidualScript} WX_SEG_COLLECT=1 WX_SEG_DIR=${WX_SEG_DIR:-$HOME/wx/segdata}
         CB=install_segdata_cb; N=${WX_N:-256}; SD=${SEED:-3} ;;
  wsegdata) export BITIDY_SCRIPT=${BITIDY_SCRIPT:-residual:ResidualScript} WX_WSEG_COLLECT=1 WX_CAMS=1 WX_SEG_DIR=${WX_SEG_DIR:-$HOME/wx/wsegdata}
         CB=install_wsegdata_cb; N=${WX_N:-256}; SD=${SEED:-3} ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
# the student without images needs no cameras
if [ "$MODE" = evalc ] || [ "$MODE" = dagger ]; then
  if [ "${WX_IMG:-1}" = 0 ]; then export WX_CAMS=${WX_CAMS:-0}; fi
  if [ "$MODE" = evalc ] && [ -z "$WX_CAMS" ]; then
    export WX_CAMS=$(cd ~ && ~/IsaacLabTutorial/.venv/bin/python -c "import torch,sys;print(int(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['cfg']['use_img']))" "$WX_STUDENT")
  fi
fi
P=newton_mjwarp
if [ "${WX_CAMS:-1}" = 1 ] || [ "${WX_TOP:-0}" = 1 ] || { [ "${WX_REAL:-0}" = 1 ] && [ "${WX_SYNTH:-0}" != 1 ]; } || [ "$MODE" = video ] || [ "$MODE" = inputs ]; then P=newton_mjwarp,newton_renderer; fi
LOG=~/wx/${MODE}_$TAG.log
echo "WX_RUN mode=$MODE tag=$TAG task=$TASK n=$N seed=$SD presets=$P cams=${WX_CAMS:-1} script=$BITIDY_SCRIPT student=$WX_STUDENT bcrnn=$BITIDY_BCRNN residual=$BITIDY_RESIDUAL $(date)" > $LOG
uv run isaaclab play --rl_library rsl_rl --task $TASK --num_envs $N --seed $SD --checkpoint "$CK" --visualizer none \
  --external_callback isaaclab_tutorial.tasks.bimanual.$(case $CB in install_segdata_cb) echo wx_seg.install_segdata;; install_wsegdata_cb) echo wx_seg.install_wsegdata;; install_fsac) echo wx_fsac.install_fsac;; *) echo wx_student.$CB;; esac) presets=$P >> $LOG 2>&1
echo "WX_EXIT=$? $(date)" >> $LOG
