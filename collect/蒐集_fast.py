# v0.0.2
# Changelog:
# - 高速資料收集版（單場資料量≈原本 3~5 倍）
# - 高速/近擊球必存，NONE 幾乎不存
# - 每次啟動自動新 CSV（利於多次啟動疊加）
# - 固定亂數種子，加快模型收斂

import os
import csv
import random
import time

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        random.seed(42)

        self.side = ai_name
        base_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # 🔥 每次啟動一個新 CSV（非常重要）
        ts = int(time.time())
        self.csv_path = os.path.join(self.data_dir, f"pingpong_fast_{ts}.csv")

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ball_x", "ball_y",
                "speed_x", "speed_y",
                "time_to_hit",
                "pred_x",
                "my_x",
                "blocker_x", "blocker_y",
                "action"
            ])

        self.last_ball_x = None
        self.last_ball_y = None

    def _predict_x(self, bx, by, vx, vy):
        if vy == 0:
            return 100
        L, R = 0, 200

        if self.side == "1P":
            target_y = 420
            if vy < 0:
                return 100
        else:
            target_y = 80
            if vy > 0:
                return 100

        t = (target_y - by) / vy
        if t < 0:
            return 100

        px = bx + vx * t
        while px < L or px > R:
            if px > R:
                px = 2 * R - px
            elif px < L:
                px = -px
        return px

    def update(self, scene_info, keyboard=[], *args, **kwargs):
        if scene_info.get("status") != "GAME_ALIVE":
            return "RESET"

        bx, by = scene_info["ball"]
        my_x = (scene_info["platform_1P"] if self.side == "1P"
                else scene_info["platform_2P"])[0]
        my_cx = my_x + 20

        blocker_x, blocker_y = -100, -100
        if scene_info.get("blocker"):
            blocker_x, blocker_y = scene_info["blocker"]

        if self.last_ball_x is None:
            vx, vy = 0, 0
        else:
            vx = bx - self.last_ball_x
            vy = by - self.last_ball_y

        self.last_ball_x, self.last_ball_y = bx, by

        if vx == 0 and vy == 0:
            return "SERVE_TO_RIGHT"

        time_to_hit = 999
        if self.side == "1P" and vy > 0:
            time_to_hit = (420 - by) / vy
        elif self.side == "2P" and vy < 0:
            time_to_hit = (by - 80) / abs(vy)

        pred_x = self._predict_x(bx, by, vx, vy)

        # === label ===
        action = "NONE"
        if my_cx < pred_x - 4:
            action = "MOVE_RIGHT"
        elif my_cx > pred_x + 4:
            action = "MOVE_LEFT"

        # === 是否保存（關鍵加速點）===
        is_high_speed = abs(vy) >= 7
        is_close = time_to_hit < 18

        should_save = False
        if is_high_speed or is_close:
            should_save = True
        elif action != "NONE" and random.random() < 0.3:
            should_save = True

        if should_save and action != "NONE":
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([bx, by, vx, vy, time_to_hit,
                            pred_x, my_x, blocker_x, blocker_y, action])

        # === 執行動作 ===
        exec_x = pred_x
        if is_high_speed:
            exec_x += random.randint(-10, 10)

        if my_cx < exec_x - 5:
            return "MOVE_RIGHT"
        if my_cx > exec_x + 5:
            return "MOVE_LEFT"
        return "NONE"

    def reset(self):
        self.last_ball_x = None
        self.last_ball_y = None
