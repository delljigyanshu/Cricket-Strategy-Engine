import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from simulator import EmpiricalSimulator
from env import CricketEnv
from utils import ensure_dir


def train(env, total_timesteps=200_000, model_path='models/ppo_cricket'):
    ensure_dir(os.path.dirname(model_path))
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=1,
        policy_kwargs={'net_arch': [256, 128]},
        batch_size=64
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    print(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed/deliveries_processed.parquet')
    parser.add_argument('--empirical', default='processed/empirical_tables.json')
    parser.add_argument('--timesteps', type=int, default=200000)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    sim = EmpiricalSimulator(df, args.empirical)
    env = CricketEnv(sim)

    train(env, total_timesteps=args.timesteps)
