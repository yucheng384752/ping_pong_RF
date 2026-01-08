# v0.0.3
# Changelog:
# - data 與 collect 同層時（../data）自動偵測並讀取
# - 其餘功能不變：自動輸出 PNG 到 analysis_result/

import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

FEATURE_COLS = [
    "ball_x", "ball_y",
    "speed_x", "speed_y",
    "time_to_hit",
    "pred_x",
    "my_x",
    "blocker_x", "blocker_y"
]
LABEL_COL = "action"
ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "NONE"]


def load_all_csv(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"找不到任何 CSV：{data_dir}\\*.csv")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("CSV 皆無法讀取或為空")

    return pd.concat(dfs, ignore_index=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = set(FEATURE_COLS + [LABEL_COL])
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV 欄位缺失: {sorted(list(miss))}")

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    df = df[df[LABEL_COL].isin(ACTIONS)].copy()
    return df


def _save_or_show(fig, out_path: str, show: bool):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    # --- 路徑自動偵測：collect/data 找不到就用 ../data ---
    script_dir = os.path.dirname(__file__)
    candidate1 = os.path.join(script_dir, "data")                  # collect/data
    candidate2 = os.path.join(os.path.dirname(script_dir), "data") # ../data (同層 data)
    default_data_dir = candidate1 if os.path.isdir(candidate1) else candidate2

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--outdir", type=str, default=os.path.join(script_dir, "analysis_result"))
    parser.add_argument("--show", action="store_true", help="同時顯示視窗（預設不顯示，只存檔）")
    parser.add_argument("--prefix", type=str, default="analysis", help="輸出檔名前綴")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=== Training Analysis (save mode) ===")
    print("data_dir:", args.data_dir)
    print("outdir  :", args.outdir)

    df = load_all_csv(args.data_dir)
    print("raw rows :", len(df))

    df = clean_df(df)
    print("clean rows:", len(df))

    # 1) |speed_y| distribution
    vy_abs = np.abs(df["speed_y"].to_numpy())

    fig = plt.figure()
    plt.hist(vy_abs, bins=30)
    plt.axvline(5, linestyle="--")
    plt.axvline(8, linestyle="--")
    plt.title("|speed_y| distribution")
    plt.xlabel("|speed_y|")
    plt.ylabel("count")
    _save_or_show(fig, os.path.join(args.outdir, f"{args.prefix}_{ts}_01_speedy_hist.png"), args.show)

    # 2) action distribution
    fig = plt.figure()
    df["action"].value_counts().plot(kind="bar")
    plt.title("Action distribution")
    plt.ylabel("count")
    _save_or_show(fig, os.path.join(args.outdir, f"{args.prefix}_{ts}_02_action_bar.png"), args.show)

    # 3) time_to_hit vs |speed_y| by action
    fig = plt.figure()
    for a in ACTIONS:
        sub = df[df["action"] == a]
        plt.scatter(sub["time_to_hit"], np.abs(sub["speed_y"]), s=6, label=a)
    plt.xlabel("time_to_hit")
    plt.ylabel("|speed_y|")
    plt.title("time_to_hit vs |speed_y| by action")
    plt.legend()
    _save_or_show(fig, os.path.join(args.outdir, f"{args.prefix}_{ts}_03_tth_vs_speedy.png"), args.show)

    # 4) temp models for analysis
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy()
    y_reg = df["pred_x"].to_numpy(dtype=np.float32)

    X_tr, X_te, y_tr, y_te, yreg_tr, yreg_te = train_test_split(
        X, y, y_reg,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample"
    )
    reg = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42
    )

    clf.fit(X_tr, y_tr)
    reg.fit(X_tr, yreg_tr)

    # 5) confusion matrix
    cm = confusion_matrix(y_te, clf.predict(X_te), labels=ACTIONS)

    fig = plt.figure()
    plt.imshow(cm)
    plt.xticks(range(len(ACTIONS)), ACTIONS)
    plt.yticks(range(len(ACTIONS)), ACTIONS)
    plt.title("Classifier Confusion Matrix")
    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    _save_or_show(fig, os.path.join(args.outdir, f"{args.prefix}_{ts}_04_confusion_matrix.png"), args.show)

    # 6) reg error vs speed
    pred = reg.predict(X_te)
    err = np.abs(pred - yreg_te)

    fig = plt.figure()
    plt.scatter(np.abs(X_te[:, 3]), err, s=6)  # speed_y index=3
    plt.xlabel("|speed_y|")
    plt.ylabel("abs(pred_x error)")
    plt.title("Regressor error vs speed")
    _save_or_show(fig, os.path.join(args.outdir, f"{args.prefix}_{ts}_05_reg_error_vs_speed.png"), args.show)

    print("Saved figures to:", args.outdir)
    print("=== Analysis finished ===")


if __name__ == "__main__":
    main()
