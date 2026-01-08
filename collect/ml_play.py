# v0.0.2
# Changelog:
# - 新增「動態 tolerance（安全帶）」：低速提早動，避免來不及接球
# - 保留原有 reg/clf gating 與 rule-based fallback
# - 不影響高速表現與 FPS

import os
import pickle
import numpy as np

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        print(f"[MLPlay] Initializing {ai_name}")

        self.dir_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.dir_path, "model.pkl")

        self.feature_cols = None
        self.clf = None
        self.reg = None
        self.legacy_model = None

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    obj = pickle.load(f)

                # 新版：dict payload
                if isinstance(obj, dict) and ("clf" in obj or "reg" in obj):
                    self.feature_cols = obj.get("feature_cols", None)
                    self.clf = obj.get("clf", None)
                    self.reg = obj.get("reg", None)
                    print("[MLPlay] Payload model loaded (clf/reg).")
                else:
                    # 舊版：單一 sklearn model
                    self.legacy_model = obj
                    print("[MLPlay] Legacy model loaded.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}")
        else:
            print("[Error] Model not found! Please run train_model.py first.")

        self.last_ball_x = None
        self.last_ball_y = None

    def predict_landing_point(self, ball_x, ball_y, speed_x, speed_y):
        if speed_y == 0:
            return 100

        bound_right = 200
        bound_left = 0

        if self.side == "1P":
            target_y = 420
            if speed_y < 0:
                return 100
        else:
            target_y = 80
            if speed_y > 0:
                return 100

        distance_y = target_y - ball_y
        time_to_hit = distance_y / speed_y
        if time_to_hit < 0:
            return 100

        pred_x = ball_x + speed_x * time_to_hit

        while pred_x < bound_left or pred_x > bound_right:
            if pred_x > bound_right:
                pred_x = 2 * bound_right - pred_x
            elif pred_x < bound_left:
                pred_x = -pred_x

        return pred_x

    def _get_blocker(self, scene_info):
        if "blocker" in scene_info and scene_info["blocker"]:
            return scene_info["blocker"][0], scene_info["blocker"][1]
        return -100, -100

    def _rule_action_from_target(self, my_center_x, target_x, tol):
        if my_center_x < target_x - tol:
            return "MOVE_RIGHT"
        if my_center_x > target_x + tol:
            return "MOVE_LEFT"
        return "NONE"

    def update(self, scene_info, keyboard=[], *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        ball_x, ball_y = scene_info["ball"]
        my_pos = scene_info["platform_1P"] if self.side == "1P" else scene_info["platform_2P"]
        my_x = my_pos[0]
        my_center_x = my_x + 20

        blocker_x, blocker_y = self._get_blocker(scene_info)

        if self.last_ball_x is None:
            speed_x, speed_y = 0, 0
        else:
            speed_x = ball_x - self.last_ball_x
            speed_y = ball_y - self.last_ball_y

        self.last_ball_x, self.last_ball_y = ball_x, ball_y

        if speed_x == 0 and speed_y == 0:
            return "SERVE_TO_RIGHT"

        # -------------------------------
        # 動態安全帶（核心修正）
        # -------------------------------
        abs_vy = abs(speed_y)
        if abs_vy <= 5:
            tol = 2      # 低速：提早動
        elif abs_vy <= 8:
            tol = 3
        else:
            tol = 4      # 高速：保留容錯

        # 1) 回中（球遠離我方時）
        ball_going_away = (self.side == "1P" and speed_y < 0) or \
                          (self.side == "2P" and speed_y > 0)
        if ball_going_away:
            if my_center_x < 90:
                return "MOVE_RIGHT"
            if my_center_x > 110:
                return "MOVE_LEFT"
            return "NONE"

        # 2) time_to_hit / pred_x（fallback 一定可用）
        time_to_hit = 999
        if self.side == "1P" and speed_y > 0:
            time_to_hit = (420 - ball_y) / speed_y
        elif self.side == "2P" and speed_y < 0:
            time_to_hit = (ball_y - 80) / abs(speed_y)

        pred_x = self.predict_landing_point(ball_x, ball_y, speed_x, speed_y)

        is_high_speed = abs_vy >= 7
        is_close_to_hit = (self.side == "1P" and (420 - ball_y) < 120) or \
                          (self.side == "2P" and (ball_y - 80) < 120)

        # 新版 payload（clf/reg）
        if self.feature_cols and (self.clf or self.reg):
            x = np.array([[ball_x, ball_y, speed_x, speed_y,
                           time_to_hit, pred_x, my_x,
                           blocker_x, blocker_y]], dtype=np.float32)

            # 高速 / 近擊球：reg 優先
            if (is_high_speed or is_close_to_hit) and self.reg is not None:
                target_x = float(self.reg.predict(x)[0])
                action = self._rule_action_from_target(my_center_x, target_x, tol)
                if action == "NONE":
                    action = self._rule_action_from_target(my_center_x, pred_x, tol)
                return action

            # 一般情況：classifier
            if self.clf is not None:
                action = self.clf.predict(x)[0]
                if action == "NONE":
                    action = self._rule_action_from_target(my_center_x, pred_x, tol)
                return action

        # 舊版單模型（向後相容）
        if self.legacy_model is not None:
            x = np.array([[ball_x, ball_y, speed_x, speed_y,
                           time_to_hit, pred_x, my_x,
                           blocker_x, blocker_y]], dtype=np.float32)
            try:
                action = self.legacy_model.predict(x)[0]
            except Exception:
                action = "NONE"

            if action == "NONE":
                action = self._rule_action_from_target(my_center_x, pred_x, tol)
            return action

        # 無模型：純規則
        return self._rule_action_from_target(my_center_x, pred_x, tol)

    def reset(self):
        self.last_ball_x = None
        self.last_ball_y = None
