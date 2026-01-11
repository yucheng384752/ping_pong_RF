# Machine Learning of Ping Pong (PAIA + Random Forest)


本專案基於 **PAIA Desktop 平台**，針對 **乒乓球對戰遊戲** 建立一套  
以 **Random Forest（Classifier + Regressor）** 為核心的 ML pipeline，目標在 **FPS=30** 下：

- 對打速度穩定（>30）
- 高速球情境下仍具備高準度
- 採用「低誤差、高容錯」訓練與推論策略（rule fallback + dynamic tolerance）
- 支援快速資料蒐集 → 快速訓練 → 立即對戰驗證

---

## 1. Design Goals

- 使用 **PAIA Desktop** 進行資料蒐集與對戰
- 使用 **Random Forest（Classifier + Regressor）**
- 強化 **hard / 高速擊球** 情境表現
- 訓練與推論流程分離、可重現
- 不依賴 GPU / torch（CPU 即可完成訓練與推論）

---

## 2. Project Structure

> 重要：目前專案的 `data/` 與 `collect/` **在同一層**（同層資料夾）

```
PINGPONG2/
│
├─ collect/
│  ├─ 蒐集.py                 # 標準資料蒐集器（PAIA Desktop）
│  ├─ 蒐集_fast.py            # 高速資料蒐集器（建議）
│  ├─ ml_play.py              # 對戰 AI（讀取 model.pkl）
│  ├─ train_fast.py           # 快速訓練腳本（輸出 model.pkl）
│  ├─ analyze_training.py     # 訓練分析（自動輸出 PNG）
│  ├─ analysis_result/        # analyze_training.py 的輸出（png）
│  └─ model.pkl               # 訓練完成後產生的模型（建議不提交 git）
│
├─ data/
│  ├─ pingpong_fast_*.csv      # 多場收集資料（建議不提交 git）
│  └─ pingpong_dataset.csv     # 其他資料（建議不提交 git）
│
├─ project.json                # PAIA Desktop 專案設定
└─ README.md
```

---

## 3. Dataset Schema (CSV)

所有訓練資料必須符合以下欄位（欄位名稱必須一致；順序可不同）：

```text
ball_x, ball_y,
speed_x, speed_y,
time_to_hit,
pred_x,
my_x,
blocker_x, blocker_y,
action
```

| 欄位 | 說明 |
|---|---|
| ball_x, ball_y | 球目前座標 |
| speed_x, speed_y | 球速度（frame diff） |
| time_to_hit | 球到達我方平台的預估時間（球非朝我方時常為 999/1000） |
| pred_x | 預測落點 x（幾何反彈估計） |
| my_x | 我方平台左上角 x |
| blocker_x, blocker_y | 障礙物座標（無則為 -100） |
| action | MOVE_LEFT / MOVE_RIGHT / NONE |

---

## 4. Data Collection (PAIA Desktop)

### 4.1 Standard Collector
- 檔案：`collect/蒐集.py`
- 特性：
  - 高速球優先保存
  - 一個 CSV 累積資料

### 4.2 Recommended: Fast Collector
- 檔案：`collect/蒐集_fast.py`
- 特性：
  - 高速球 + 近擊球區 **100% 保存**
  - NONE 樣本大幅減少
  - 每次啟動自動產生新 CSV（利於多次啟動累積）
  - 單場資料量約為標準版的 **3~5 倍**

PAIA Desktop 中請選擇 AI 腳本：
```
collect/蒐集_fast.py
```

---

## 5. Training (Random Forest Dual-Head)

### 5.1 Trainer Script
- 檔案：`collect/train_fast.py`
- 功能：
  - 自動合併 `data/*.csv`（多 CSV 合併訓練）
  - 自動清洗資料（含 `time_to_hit` 雜訊過濾）
  - hard case 加權（提升高速表現）
  - 同時訓練：
    - `RandomForestClassifier`：預測 action
    - `RandomForestRegressor`：預測 target_x（落點）

### 5.2 Default Cleaning Policy (High-ROI)
為了提升「可決策 frame」比例與容錯性，訓練預設會做：

- 過濾 `time_to_hit >= 900`（移除 999/1000 雜訊樣本）
- 丟掉 `action==NONE` 且 `time_to_hit > 40`（避免模型過度保守）

> 若你確定要保留遠端樣本，可調整參數（見下方 CLI）。

### 5.3 Train Commands

在專案根目錄執行：

**(A) 最終訓練（全量）**
```bash
python collect/train_fast.py
```

**(B) 快速迭代（抽樣 50,000 筆）**
```bash
python collect/train_fast.py --sample_n 50000
```

**(C) 調整 time_to_hit 過濾**
```bash
python collect/train_fast.py --tth_max_keep 999
```

訓練輸出範例：
```text
raw rows : 85070
clean rows: 42918
[Stats] rows=42918  NONE=0.72%  |vy|>=9=73.54%  |vy|<=5=1.37%
[Classifier] acc=0.9895
[Regressor ] pred_x MAE=0.582
Saved: collect/model.pkl
```

---

## 6. Play (Using model.pkl)

### 6.1 Player Script
- 檔案：`collect/ml_play.py`
- 行為：
  - 自動載入同層 `model.pkl`
  - 兼容：
    - 舊版單模型 `.pkl`
    - 新版 dict payload（`feature_cols + clf + reg + meta`）
  - 高速/近擊球：regressor 優先（連續、穩）
  - 一般情況：classifier
  - 永遠保留 rule-based fallback（高容錯）
  - 低速時採用 **dynamic tolerance**（避免慢球來不及接）

PAIA Desktop 中請選擇 AI 腳本：
```
collect/ml_play.py
```

---

## 7. Training Visualization (Auto Save PNG)

- 檔案：`collect/analyze_training.py`
- 自動輸出圖檔到：`collect/analysis_result/`

使用方式：
```bash
python collect/analyze_training.py
```

可選：同時顯示視窗
```bash
python collect/analyze_training.py --show
```

**建議重點觀察：**
1) `|speed_y|` 分佈（低速/高速比例是否合理）  
2) action 分佈（NONE 是否過多/過少）  
3) `time_to_hit vs |speed_y|`（是否仍有大量 999/1000 右側直線）  
4) confusion matrix（MOVE 誤判 NONE 的比例）  
5) regressor error vs speed（高速區是否有尖峰）

---

## 8. Recommended Workflow (Best Practice)

1. PAIA Desktop 用 `蒐集_fast.py` 收資料 2–3 分鐘（多跑幾場）
2. 執行 `train_fast.py`（先 sample_n 快速迭代，再全量最終訓練）
3. 執行 `analyze_training.py` 產出分析圖
4. PAIA Desktop 用 `ml_play.py` 對戰（hard / normal）

---

## 9. PAIA MLPlay Agent API Specification

> 本章節定義專案內 AI Agent（Collector / Player）與 PAIA Engine 之介面契約。

### 9.1 MLPlay Class Interface

```python
class MLPlay:
    def __init__(self, ai_name: str, *args, **kwargs):
        ...

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs) -> str:
        ...

    def reset(self):
        ...
```

- `ai_name`: `"1P"` 或 `"2P"`
- `update()` 回傳 action 字串（見 9.3）
- 當 `scene_info["status"] != "GAME_ALIVE"` 時回傳 `"RESET"`

### 9.2 scene_info Input Contract

```python
scene_info = {
    "status": "GAME_ALIVE" | "GAME_OVER" | ...,
    "ball": [ball_x, ball_y],
    "platform_1P": [x, y],
    "platform_2P": [x, y],
    "blocker": [x, y] | None
}
```

### 9.3 action Output Contract

`update()` 必須回傳以下字串之一：

```text
"MOVE_LEFT" | "MOVE_RIGHT" | "NONE" | "RESET" | "SERVE_TO_RIGHT"
```

### 9.4 Model Payload Contract (model.pkl)

`collect/train_fast.py` 輸出 `model.pkl` 建議為 dict payload：

```python
payload = {
    "feature_cols": [...],     # List[str]
    "clf": RandomForestClassifier,
    "reg": RandomForestRegressor,
    "meta": {
        "version": str,
        "rows_used": int,
        "tth_max_keep": float | None,
        "sample_n": int | None,
        "seed": int,
        "test_size": float,
    }
}
```

`collect/ml_play.py` 需維持向後相容：若載入不是 dict，視為 legacy 單模型。

---

## 10. Environment

- Python 3.9+
- numpy, pandas, scikit-learn
- PAIA Desktop
- 不需要 torch / CUDA（本專案 RF 推論/訓練為 CPU 即可）

---

## 11. Git Upload Notes

建議不要提交：
- `.venv/`
- `data/*.csv`
- `collect/model.pkl`
- `collect/analysis_result/*.png`

建議 `.gitignore` 包含：
```gitignore
.venv/
__pycache__/
*.csv
*.pkl
analysis_result/
*.log
```
