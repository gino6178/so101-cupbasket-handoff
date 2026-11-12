#!/bin/bash
# wx_setup.sh: unpack the Isaac Lab install copied from h1 (IsaacLabTutorial + uv python), then a CupBasket dummy checkpoint
exec >> ~/wx_setup.log 2>&1
set -x
cd /home/Gino
if [ -f wx_isaac.tar.zst ]; then tar -I zstd -xf wx_isaac.tar.zst && rm -f wx_isaac.tar.zst; fi
mkdir -p ~/bimanual ~/wx ~/IsaacLabTutorial/src/isaaclab_tutorial/tasks/bimanual
export PATH=$HOME/.local/bin:$PATH
uv --version
bash ~/cb_dummy.sh
echo WX_SETUP_DONE
