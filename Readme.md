# Machine Learning of Ping Pong (PAIA + Random Forest)

本專案基於 **PAIA Desktop 平台**，針對 **乒乓球對戰遊戲** 設計一套  
以 **Random Forest** 為核心的機器學習流程，目標是在 **FPS = 30** 條件下：

- 對打速度穩定（> 30）
- 高速球情境下仍具備高準度
- 採用「低誤差、高容錯」的模型訓練策略
- 支援快速資料蒐集與快速模型迭代

---

## 一、專案目標（Design Goals）

- 使用 **PAIA Desktop** 進行資料蒐集與對戰
- 使用 **Random Forest（Classifier + Regressor）**
- 強化 **hard / 高速擊球** 情境表現
- 訓練與推論流程完全分離、可重現
- 不依賴 GPU / torch（CPU 即可完成訓練與推論）

---

## 二、專案結構說明

PINGPONG2/
│
├─ collect/
│ ├─ 蒐集.py # 標準資料蒐集器（PAIA Desktop 使用）
│ ├─ 蒐集_fast.py # 高速資料蒐集器（建議使用）
│ ├─ ml_play.py # 對戰 AI（讀取 model.pkl）
│ ├─ train_model_fast.py # 快速訓練腳本（輸出 model.pkl）
│ ├─ model.pkl # 訓練完成後產生的模型
│ └─ data/
│ ├─ pingpong_dataset.csv
│ └─ pingpong_fast_*.csv
│
├─ project.json # PAIA Desktop 專案設定
└─ README.md

---

## 三、資料格式（CSV Schema）

所有訓練資料必須符合以下欄位（順序不影響，但名稱必須一致）：

```text
ball_x, ball_y,
speed_x, speed_y,
time_to_hit,
pred_x,
my_x,
blocker_x, blocker_y,
action
```

| 欄位                   | 說明                            |
| -------------------- | ----------------------------- |
| ball_x, ball_y       | 球目前座標                         |
| speed_x, speed_y     | 球速度（frame 差分）                 |
| time_to_hit          | 球到達我方平台的預估時間                  |
| pred_x               | 預測落點 x                        |
| my_x                 | 我方平台左上角 x                     |
| blocker_x, blocker_y | 障礙物座標（無則為 -100）               |
| action               | MOVE_LEFT / MOVE_RIGHT / NONE |

## 四、資料蒐集（PAIA Desktop）

標準蒐集

- 使用：collect/蒐集.py
- 特性：
  - 高速球優先保存
  - 一個 CSV 累積資料

建議使用：高速蒐集
- 使用：collect/蒐集_fast.py
- 特性：
  - 高速球 + 近擊球區 100% 保存
  - NONE 樣本大幅減少
  - 每次啟動自動產生新 CSV
  - 單場資料量約為標準版的 3~5 倍
  
在 PAIA Desktop 中選擇 AI 腳本為：collect/蒐集_fast.py

## 五、快速訓練（Random Forest）

使用腳本:
`collect/train_model_fast.py`

特點
- 支援 data/*.csv 多 CSV 合併訓練
- 自動資料清洗
- 高速樣本加權（提升 hard 情境準度）
- 同時訓練：
  - RandomForestClassifier（預測 action）
  - RandomForestRegressor（預測落點 x）
- 輸出：
  - `collect/model.pkl`
- 訓練輸出範例
  ```yaml
  raw rows: 2571
  clean rows: 2426
  [Classifier] acc=0.93
  [Regressor ] pred_x MAE=1.52
  Saved: model.pkl
  ```

## 六、對戰（使用訓練好的模型）
對戰 AI
```bash
collect/ml_play.py
```
特性
- 自動載入同層 model.pkl
- 支援：
    - 舊版單模型 .pkl
    - 新版 dict payload（clf + reg）
- 高速 / 近擊球：
    - 優先使用 regressor（連續、穩定）
- 一般情境：
    - 使用 classifier
- 永遠保留 rule-based fallback（高容錯）

## 七、建議工作流程（Best Practice）
快速迭代流程
1. 使用 蒐集_fast.py 收資料 2–3 分鐘
2. 執行 train_model_fast.py
3. 使用 ml_play.py 對戰測試
4. 重複 2–3 次

**通常 第 2–3 輪模型即可穩定壓過內建（hard**

## 八、環境需求

- Python 3.9+
- numpy, pandas, scikit-learn
- PAIA Desktop