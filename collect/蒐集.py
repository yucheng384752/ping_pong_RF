# v0.0.1
# Changelog:
# - PAIA Desktop collector: 產生 pingpong 訓練資料 CSV（含 blocker + time_to_hit）
# - 高速樣本優先保存，提升 hard 場景資料密度

import os
import csv
import random

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name  # "1P" or "2P"

        base_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.csv_path = os.path.join(self.data_dir, "pingpong_dataset.csv")

        # header 固定（務必與 train_model / ml_play 推論一致）
        if (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0):
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

    def _predict_landing_x(self, ball_x, ball_y, vx, vy):
        if vy == 0:
            return 100

        # PAIA pingpong 邊界通常是 0~200
        L, R = 0, 200

        if self.side == "1P":
            target_y = 420
            if vy < 0:
                return 100
        else:
            target_y = 80
            if vy > 0:
                return 100

        t = (target_y - ball_y) / vy
        if t < 0:
            return 100

        pred_x = ball_x + vx * t

        # 牆反彈
        while pred_x < L or pred_x > R:
            if pred_x > R:
                pred_x = 2 * R - pred_x
            elif pred_x < L:
                pred_x = -pred_x

        return pred_x

    def update(self, scene_info, keyboard=[], *args, **kwargs):
        # PAIA 的狀態 key 通常是 "status"
        if scene_info.get("status") != "GAME_ALIVE":
            return "RESET"

        ball_x, ball_y = scene_info["ball"][0], scene_info["ball"][1]

        # 平台座標
        my_pos = scene_info["platform_1P"] if self.side == "1P" else scene_info["platform_2P"]
        my_x = my_pos[0]
        my_center_x = my_x + 20

        # blocker（有些場景沒有）
        blocker_x, blocker_y = -100, -100
        if "blocker" in scene_info and scene_info["blocker"]:
            blocker_x, blocker_y = scene_info["blocker"][0], scene_info["blocker"][1]

        # 速度
        if self.last_ball_x is None:
            vx, vy = 0, 0
        else:
            vx = ball_x - self.last_ball_x
            vy = ball_y - self.last_ball_y

        self.last_ball_x, self.last_ball_y = ball_x, ball_y

        # 發球
        if vx == 0 and vy == 0:
            return random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

        # time_to_hit
        time_to_hit = 999
        if self.side == "1P" and vy > 0:
            time_to_hit = (420 - ball_y) / vy
        elif self.side == "2P" and vy < 0:
            time_to_hit = (ball_y - 80) / abs(vy)

        pred_x = self._predict_landing_x(ball_x, ball_y, vx, vy)

        # 收集 label（偏向「應該怎麼走」）
        action_save = "NONE"
        if my_center_x < pred_x - 4:
            action_save = "MOVE_RIGHT"
        elif my_center_x > pred_x + 4:
            action_save = "MOVE_LEFT"

        # 執行動作（可有 jitter 模擬容錯）
        is_high_speed = abs(vy) > 7
        target_exec = pred_x
        if is_high_speed:
            target_exec += random.randint(-15, 15)

        action_exec = "NONE"
        if my_center_x < target_exec - 5:
            action_exec = "MOVE_RIGHT"
        elif my_center_x > target_exec + 5:
            action_exec = "MOVE_LEFT"

        # 決定是否保存（高速優先）
        should_save = False
        if is_high_speed:
            should_save = True
        elif action_save != "NONE":
            should_save = (random.random() < 0.3)
        else:
            should_save = (random.random() < 0.1)

        if should_save:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ball_x, ball_y, vx, vy, time_to_hit, pred_x, my_x, blocker_x, blocker_y, action_save])

        return action_exec

    def reset(self):
        self.last_ball_x = None
        self.last_ball_y = None
