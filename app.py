from flask import Flask, render_template, request, jsonify, current_app
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from simulator import EmpiricalSimulator
from env import CricketEnv
import threading
import traceback
import os

app = Flask(__name__)

DF_PATH = "processed/deliveries_processed.parquet"
EMP_JSON = "processed/empirical_tables.json"
MODEL_PATH = "models/ppo_cricket.zip"
MAPPINGS = "processed/mappings.json"

if not os.path.exists(DF_PATH):
    raise FileNotFoundError(f"{DF_PATH} missing")

if not os.path.exists(EMP_JSON):
    raise FileNotFoundError(f"{EMP_JSON} missing")

df = pd.read_parquet(DF_PATH)
sim = EmpiricalSimulator(df, EMP_JSON, mappings_path=MAPPINGS)
env = CricketEnv(sim)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} missing")
model = PPO.load(MODEL_PATH)

_simulate_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/simulate_ajax", methods=["POST"])
def simulate_ajax():
    acquired = _simulate_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "busy", "message": "Simulation already running. Please wait."}), 409

    try:
        body = request.get_json(silent=True) or {}
        batting_team = body.get("batting_team", "TeamA")
        batting_order = body.get("batting_order", None)
        sim.reset_match(batting_team_name=batting_team,
                        batting_order=batting_order if isinstance(batting_order, list) else None)

        res = env.reset()
        obs = res[0] if isinstance(res, (tuple, list)) and len(res) >= 1 else res

        done = False
        ball_num = 1

        lines = []
        balls = []
        scores = []
        wickets_arr = []
        bowlers_count = {}

        while not done:
            try:
                obs_arr = np.asarray(obs, dtype=np.float32)
            except Exception:
                obs_arr = np.asarray(obs)

            try:
                action, _ = model.predict(obs_arr, deterministic=True)
            except Exception:
                action, _ = model.predict(np.asarray([obs_arr]), deterministic=True)
                if getattr(action, "shape", None) and action.shape[0] == 1:
                    action = action[0]

            try:
                a0 = int(np.ravel(action)[0])
                a1 = int(np.ravel(action)[1])
            except Exception:
                a0, a1 = 0, 1

            res2 = env.step([a0, a1])

            info = {}
            reward = 0
            if isinstance(res2, (tuple, list)) and len(res2) >= 4:
                obs = res2[0]
                reward = res2[1]
                terminated = bool(res2[2])
                truncated = bool(res2[3]) if len(res2) > 3 else False
                done = terminated or truncated
                if len(res2) > 4:
                    info = res2[4] or {}
            elif isinstance(res2, (tuple, list)) and len(res2) == 3:
                obs = res2[0]
                reward = res2[1]
                done = bool(res2[2])
            else:
                obs = res2
                done = False

            outcome = info.get("outcome", {}) if isinstance(info, dict) else {}
            bowler_used = info.get("bowler_used") if isinstance(info, dict) else None

            if bowler_used:
                bowler_name = bowler_used
            elif outcome.get("bowler"):
                bowler_name = outcome.get("bowler")
            else:
                bowler_name = f"bowler_{a0}"

            batsman_name = outcome.get("batsman", getattr(sim, "current_batsman", "batsman_0"))
            runs = int(outcome.get("runs", 0)) if outcome else 0
            wicket = bool(outcome.get("wicket", False)) if outcome else False

            bowlers_count[bowler_name] = bowlers_count.get(bowler_name, 0) + 1
            lines.append({
                "ball": int(ball_num),
                "bowler_id": f"bowler_{a0}",
                "bowler_name": str(bowler_name),
                "batsman": str(batsman_name),
                "intent": {0: "defensive", 1: "normal", 2: "aggressive"}.get(int(a1), "normal"),
                "runs": int(runs),
                "wicket": bool(wicket),
                "score": f"{int(sim.score)}/{int(sim.wickets)}"
            })

            balls.append(int(ball_num))
            scores.append(int(sim.score))
            wickets_arr.append(int(sim.wickets))

            ball_num += 1
            if ball_num > 200:
                lines.append({"note": "Stopped early: safety limit reached"})
                break

        balls_py = [int(x) for x in balls]
        scores_py = [int(x) for x in scores]
        wickets_py = [int(x) for x in wickets_arr]
        bowler_usage_py = {str(k): int(v) for k, v in bowlers_count.items()}

        lines_py = []
        for l in lines:
            if "note" in l:
                lines_py.append({"note": str(l["note"])})
                continue
            lines_py.append({
                "ball": int(l.get("ball", 0)),
                "bowler_id": str(l.get("bowler_id", "")),
                "bowler_name": str(l.get("bowler_name", "")),
                "batsman": str(l.get("batsman", "")),
                "intent": str(l.get("intent", "")),
                "runs": int(l.get("runs", 0)),
                "wicket": bool(l.get("wicket", False)),
                "score": str(l.get("score", "0/0"))
            })

        final_score = f"{int(sim.score)}/{int(sim.wickets)}"
        response = {
            "lines": lines_py,
            "final_score": final_score,
            "chart": {"balls": balls_py, "scores": scores_py, "wickets": wickets_py},
            "bowler_usage": bowler_usage_py
        }
        return jsonify(response)

    except Exception as e:
        tb = traceback.format_exc()
        current_app.logger.error("simulate_ajax error: %s", tb)
        return jsonify({"error": "server_error", "message": str(e)}), 500

    finally:
        _simulate_lock.release()


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
