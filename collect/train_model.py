# v0.0.1
# Changelog:
# - 修正欄位不一致：加入 time_to_hit，並統一 feature schema 供 ml_play 使用
# - 同時訓練 RandomForestRegressor(預測落點x) + RandomForestClassifier(預測動作)
# - 針對高速樣本加權，提升高速擊球精度
# - 模型輸出改為 dict（包含 feature_cols / models），ml_play 向後相容處理

import os
import glob
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..\\data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ---- 單一真實來源：特徵欄位順序（collector 寫入 + ml_play 推論必須一致）----
FEATURE_COLS = [
    "ball_x", "ball_y",
    "speed_x", "speed_y",
    "time_to_hit",
    "pred_x",
    "my_x",
    "blocker_x", "blocker_y"
]

LABEL_COL = "action"
REG_TARGET_COL = "pred_x"   # reg 目標：預測落點 x（用於高速更穩）
ALLOWED_ACTIONS = {"MOVE_LEFT", "MOVE_RIGHT", "NONE"}


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    # 必要欄位檢查
    needed = set(FEATURE_COLS + [LABEL_COL])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV 欄位缺失: {sorted(list(missing))}")

    # 強制轉數字（action 제외）
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = df[LABEL_COL].astype(str)

    # 移除壞資料
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()

    # 過濾 label
    df = df[df[LABEL_COL].isin(ALLOWED_ACTIONS)].copy()

    # 基本合理性過濾（避免極端噪聲）
    df = df[(df["ball_x"] >= -50) & (df["ball_x"] <= 250)]
    df = df[(df["pred_x"] >= -50) & (df["pred_x"] <= 250)]
    df = df[(df["time_to_hit"] >= 0) & (df["time_to_hit"] <= 999)]

    return df


def train():
    print("--- 開始訓練程序 (RF dual-head: reg+clf) ---")

    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        print("錯誤：找不到資料，請確認 data/*.csv 存在")
        return

    df_list = []
    for f in all_files:
        dfi = _safe_read_csv(f)
        if dfi is not None and len(dfi) > 0:
            df_list.append(dfi)

    if not df_list:
        print("錯誤：CSV 皆無法讀取或為空")
        return

    df = pd.concat(df_list, axis=0, ignore_index=True)

    print(f"清洗前筆數: {len(df)}")
    df = _clean_and_validate(df)
    print(f"清洗後筆數: {len(df)}")
    if len(df) < 5000:
        print("[警告] 資料量偏少，模型容易打不贏內建。建議先多收集對戰資料。")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy()
    y_reg = df[REG_TARGET_COL].to_numpy(dtype=np.float32)

    # 高速加權：|speed_y| 越大權重越高（強化高速精度）
    # 低誤差 / 高容錯：不追求過度擬合，使用平滑權重
    speed_y_abs = np.abs(df["speed_y"].to_numpy(dtype=np.float32))
    sample_weight = 1.0 + np.clip((speed_y_abs - 5.0) / 5.0, 0.0, 3.0)  # 1~4

    X_train, X_test, y_train, y_test, w_train, w_test, yreg_train, yreg_test = train_test_split(
        X, y, sample_weight, y_reg,
        test_size=0.2, random_state=42, stratify=y
    )

    # --- Classifier：預測動作 ---
    clf = RandomForestClassifier(
        n_estimators=220,          # 推論速度與準度平衡（FPS 30 夠用）
        max_depth=18,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    # --- Regressor：預測落點 x（高速用更穩）---
    reg = RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_train, y_train, sample_weight=w_train)
    reg.fit(X_train, yreg_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Classifier] accuracy: {acc:.4f}")

    # reg 誤差簡單檢查（越小越好）
    reg_pred = reg.predict(X_test)
    mae = float(np.mean(np.abs(reg_pred - yreg_test)))
    print(f"[Regressor] pred_x MAE: {mae:.3f}")

    payload = {
        "feature_cols": FEATURE_COLS,
        "clf": clf,
        "reg": reg,
        "meta": {
            "version": "0.0.1",
            "allowed_actions": sorted(list(ALLOWED_ACTIONS)),
        }
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"模型已儲存至: {MODEL_PATH}")
    print("訓練完成。")


if __name__ == "__main__":
    train()
