# v0.0.2
# Changelog:
# - 快速訓練版：縮短訓練時間，仍輸出 ml_play 可用的 model.pkl（dict payload）
# - 支援多 CSV 合併、清洗、以及高速樣本加權（提升 hard 精度）
# - 使用較小 RF 規模以確保推論 FPS 30 穩定

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

FEATURE_COLS = [
    "ball_x", "ball_y",
    "speed_x", "speed_y",
    "time_to_hit",
    "pred_x",
    "my_x",
    "blocker_x", "blocker_y"
]
LABEL_COL = "action"
ALLOWED_ACTIONS = {"MOVE_LEFT", "MOVE_RIGHT", "NONE"}

def _read_all_csv(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"找不到資料：{data_dir}\\*.csv")

    dfs = []
    for f in files:
        try:
            dfi = pd.read_csv(f)
            if len(dfi) > 0:
                dfs.append(dfi)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("CSV 皆無法讀取或為空")

    return pd.concat(dfs, axis=0, ignore_index=True)

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    needed = set(FEATURE_COLS + [LABEL_COL])
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV 欄位缺失: {sorted(list(miss))}")

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    df = df[df[LABEL_COL].isin(ALLOWED_ACTIONS)].copy()

    # 合理範圍（避免髒資料拖慢/拉歪）
    df = df[(df["ball_x"] >= -50) & (df["ball_x"] <= 250)]
    df = df[(df["pred_x"] >= -50) & (df["pred_x"] <= 250)]
    df = df[(df["time_to_hit"] >= 0) & (df["time_to_hit"] <= 999)]

    return df

def train():
    print("--- Fast Train (RF dual-head) ---")
    df = _read_all_csv(DATA_DIR)
    print("raw rows:", len(df))

    df = _clean(df)
    print("clean rows:", len(df))

    if len(df) < 2000:
        print("[WARN] 資料量偏少，先多收集（尤其高速）才更容易贏內建。")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy()
    y_reg = df["pred_x"].to_numpy(dtype=np.float32)

    # 高速加權（平滑、容錯）
    vy_abs = np.abs(df["speed_y"].to_numpy(dtype=np.float32))
    w = 1.0 + np.clip((vy_abs - 5.0) / 6.0, 0.0, 2.0)  # 1~3（比完整版更保守、泛化更好）

    X_tr, X_te, y_tr, y_te, w_tr, w_te, yreg_tr, yreg_te = train_test_split(
        X, y, w, y_reg, test_size=0.2, random_state=42, stratify=y
    )

    # 快速參數：樹少一點、深度合理，推論也更快
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    reg.fit(X_tr, yreg_tr, sample_weight=w_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    mae = float(np.mean(np.abs(reg.predict(X_te) - yreg_te)))

    print(f"[Classifier] acc={acc:.4f}")
    print(f"[Regressor ] pred_x MAE={mae:.3f}")

    payload = {
        "feature_cols": FEATURE_COLS,
        "clf": clf,
        "reg": reg,
        "meta": {"version": "0.0.2-fast"}
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    train()
