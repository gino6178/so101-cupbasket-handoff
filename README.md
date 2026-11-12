# SO-101 雙臂杯碗入籃：學生策略 g2 交接包

寫於 2026-11-12。交接對象是負責實機的同仁。

## 一句話現況

`checkpoints/g2.pt` 是目前最好的、只用實機拿得到的資訊的策略。
- 在模擬的真實設定下，配合「夾爪規則」成功率約 **60%**。
- **還沒有上過實機**，實機端的感知與觀測組裝程式也還沒寫（見最後一節）。

## 1. 成績（全部是模擬結果）

設定：
- `WX_REAL=1`：相機模型、延遲、編碼器量化與雜訊、標定誤差。
- 物理隨機化：PD 增益 ±30%、關節摩擦、質量、材質。
- 杯碗間距 3 cm，頂部相機姿態 G。

每格 512 回合。

| 設定 | 成功率 |
|---|---|
| g2 + 夾爪規則 + 手臂動作雜訊 0.35（三個 seed） | 62.5% / 59.8% / 61.3% |
| g2 + 夾爪規則，無雜訊（確定性） | 54–57% |
| g2 單獨（無規則，手臂雜訊 0.35） | 42–46% |
| 參考：特權老師（吃真值，實機不能用） | 81–85% |

主要失敗原因：
- **沒夾起來：** 夾爪到了杯緣卻沒下閉合指令。規則就是補這個。
- **放開後夾爪勾在杯內：** 杯子被拖走或倒下。
- **放開位置偏離目標區：** 杯碗傾倒或落在區外。

## 2. 模型輸入與輸出

控制頻率 30 Hz（模擬步長 1/120 s，decimation 4）。

模型是 LSTM 512×2（`sim/bimanual/wx_student.py` 的 `Student`），每一步吃三樣東西。

### (1) 狀態向量 s，163 維

順序依觀測組 `coarse` 的 term 排列。正規化在模型內完成，餵原始值即可。

| term | 維度 | 內容 |
|---|---|---|
| left/right_joint_pos | 6 + 6 | 關節角（5 個手臂關節 + 夾爪），rad |
| left/right_joint_vel | 6 + 6 | 關節速度：編碼器差分 |
| left/right_joint_target | 6 + 6 | 目前寫給伺服的目標角 |
| previous_action | 12 | 上一步的動作輸出 |
| pads | 6 | 兩支夾爪指墊中心位置（順運動學） |
| cube | 54 | 每個杯碗與籃子的位姿：**來自相機定位**（`wx_env.coarse_scene`） |
| relative | 18 | 每支手臂相對於目標物和目標位置的位移（`wx_env.coarse_relative`） |
| command | 20 | 每支手臂：要拿哪個物件（one-hot 6）、放哪個籃子（one-hot 3）、是否啟用（1） |
| progress | 11 | 真實設定下全部填 0 |
| base | 6 | 兩支手臂底座偏移 |

確切順序與各項定義以程式為準：
- 模擬中可印出 `env.observation_manager.active_terms["coarse"]` 確認。
- 各函式定義在 `cup_basket_env_cfg.py` / `env_cfg.py` / `wx_env.py`。

### (2) 腕部影像 ×2

- 格式：RGB uint8，320×240，形狀 (2, 3, 240, 320)，順序為左、右。
- 鏡頭模型：模擬中模擬 OV9732 約 100° 廣角加桶形畸變，見 `wx_real.WristReal`。

### (3) 頂部影像

- C920 畫面縮成 320×180 的 RGB uint8。
- 相機姿態 G：位於兩臂之間、籃子列前方。鏡頭在 (0.17, −0.95, 0.80) m，看向 (0.17, 0, 0.02)；座標定義見 `sim/SCENE_SETUP.md`。
- 注意：`SCENE_SETUP.md` 寫的是舊的姿態 A，相機位置請以本段為準。

### 輸出：12 維動作

每支手臂 6 維：5 個手臂關節 + 夾爪。左臂在前，右臂在後。

- **手臂：** 相對關節位置。新目標 = 舊目標 + 0.033 rad × 輸出，輸出截在 ±2。
- **夾爪：** 看正負號。大於 0 為張開（PREGRASP_GRIPPER_POSITION），否則為閉合（−12°）。

## 3. 夾爪規則（60% 一定要搭配）

實作是 `wx_student.py` 的 `gripper_shields()`，評估時用的參數如下：

```
WX_GRIP_BIAS=0.6 WX_REL_BIAS=0.6 WX_REL_NEAR=0.05,0.08 WX_REL_LATCH=1
```

1. **抓取偏置：** 手臂還沒拿東西時，夾爪輸出 < 0.6 就閉合。原本是 < 0。
2. **放開偏置：** 手臂拿著東西，且物件距目標水平 < 5 cm、高度差 < 8 cm 時，夾爪輸出 > −0.6 就張開。
3. **放開閂鎖：** 在目標附近放開後，這支手臂不再套用抓取偏置。

「是否拿著」和「離目標多遠」都來自相機定位（`wx_env.coarse`），實機也拿得到。

## 4. 檔案

| 路徑 | 內容 |
|---|---|
| `checkpoints/g2.pt` | 學生策略。裡面有 `cfg`、`state_dict`，以及訓練時的環境變數 `env` |
| `checkpoints/segnet_G.pt` | 頂部相機分割網路（姿態 G） |
| `checkpoints/wsegnet1.pt` | 腕部相機分割網路 |
| `sim/bimanual/` | 模擬端完整套件（Isaac Lab），與評估時的版本一致 |
| `sim/cb3_geo.env` | 場景幾何：手臂底座位置、籃子尺寸 180×120 mm、目標區 |
| `sim/wx_run.sh` | 模擬評估啟動腳本 |
| `sim/DEPLOY_PLAN.md` | 11/10 的實機部署計畫與缺口盤點 |
| `sim/SCENE_SETUP.md` | 實機場景佈置表。相機位置是舊的姿態 A，請改用第 2 節的 G |
| `real/sort_cell.py` | 舊任務的實機程式。Feetech 驅動、關節對應、30 Hz 控制迴圈、安全流程可沿用 |
| `print/` | 杯 ×3、碗 ×3、籃子 ×3 的 STL，尺寸與模擬一致；顏色與重量注意事項見 `print/README.md` |

`sim/bimanual` 依賴 `isaaclab_tutorial.tasks.place_vial`（SO-101 資產、mdp），這部分不在這個 repo 裡。

## 5. 在模擬中重現 60%

把 `sim/bimanual` 放到 `isaaclab_tutorial/tasks/bimanual`，檔案放在 `~/wx/`，然後執行：

```bash
E="BITIDY_RESIDUAL=<teacher residual> BITIDY_BCRNN=<teacher bcrnn> WX_ENVFILE=~/cb3_geo.env BITIDY_CB_PHYS=gains,joints,mass,mat BITIDY_CB_OBJ_SEP=0.03 WX_REAL=1 WX_TOP_POSE=G WX_SEGNET=~/wx/segnet_G.pt WX_TOP_Z_CUP=0.035 WX_TOP_Z_BOWL=0.03 WX_WRIST_OUT=320x240 WX_WAUG=0.5 WX_TOPIMG=1 WX_TOPIMG_WH=320x180 WX_WFUSE=1 WX_WSEGNET=~/wx/wsegnet1.pt BITIDY_CB_DIAG=1 WX_N=512 SEED=7 WX_STUDENT=~/wx/g2.pt WX_EVALC_SCRIPT=wx_student:StudentResScript WX_RRES=<any rres ckpt> WX_RRES_SCALE=0 WX_RRES_EVAL_STD=0.35,0 WX_GRIP_BIAS=0.6 WX_REL_BIAS=0.6 WX_REL_NEAR=0.05,0.08 WX_REL_LATCH=1"
bash ~/wx/wx_run.sh evalc g2_check $E
```

說明：
- `WX_RRES_SCALE=0` 表示不用殘差，只加 0.35 的手臂動作雜訊。
- 老師檔案只是環境初始化需要，並不參與學生的決策。
- 想跑確定性版本，把 `WX_RRES_EVAL_STD` 改成 `0`。

## 6. 上實機前還缺什麼

細節見 `sim/DEPLOY_PLAN.md`。

1. **感知要移植：** 分割 → 融合定位（`wx_top.py`、`wx_fuse.py`、`wx_seg.py`）目前依賴 Isaac Lab，要改成可在實機電腦上獨立執行的版本。輸出就是第 2 節的 `cube` / `relative`。
2. **分割網路沒看過真實影像：** 實機需要先拍照檢查，很可能要用實拍影像微調。
3. **相機標定：** 需要 C920 內參與外參（AprilTag + solvePnP），以及腕部相機的安裝位置與視角。
4. **觀測組裝：** 依第 2 節的 163 維順序組裝，再加上 LSTM 狀態管理（每回合開始時歸零）。
5. **夾爪規則：** 依第 3 節移植。
6. **重量差：** 列印的杯碗約 25–28 g，模擬是 8–10 g。請秤重回報，模擬端會更新。
7. **安全：** 從 home 姿態開始、預設 dry run、結束時鬆力矩。沿用 `real/sort_cell.py` 的作法。
