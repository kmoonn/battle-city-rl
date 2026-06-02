#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
继续训练脚本 - 基于现有模型继续训练
"""

import os
import sys
import glob

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battle_city_env import BattleCityEnv
from config import DQN_CONFIG, CALLBACK_CONFIG, get_log_dir, get_model_dir
from utils import setup_directories, print_training_info


def make_env(level=1):
    """创建环境的工厂函数"""
    def _init():
        env = BattleCityEnv(render_mode=None, level=level)
        return Monitor(env)
    return _init


def find_best_model(base_dir="./models"):
    """查找最佳模型"""
    # 首先查找 best_model
    best_models = sorted(glob.glob(os.path.join(base_dir, "battle_city_dqn_*/best/best_model.zip")))
    if best_models:
        return best_models[-1]

    # 然后查找 final_model
    final_models = sorted(glob.glob(os.path.join(base_dir, "battle_city_dqn_*/final_model.zip")))
    if final_models:
        return final_models[-1]

    return None


def continue_training(model_path=None, additional_steps=None):
    """基于现有模型继续训练

    Args:
        model_path: 模型路径，None 则自动查找
        additional_steps: 额外训练步数，None 则使用默认值
    """
    # 查找模型
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("错误: 未找到可用的 DQN 模型")
            print("请先运行 train_dqn.py 进行初始训练")
            return None
        print(f"自动找到模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None

    # 设置训练步数
    if additional_steps is None:
        additional_steps = DQN_CONFIG['total_timesteps']

    # 创建保存目录
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/battle_city_dqn_continue_{timestamp}"
    model_dir = f"./models/battle_city_dqn_continue_{timestamp}"
    setup_directories(log_dir, model_dir)

    print("=" * 60)
    print("继续训练 DQN 模型")
    print("=" * 60)
    print(f"加载模型: {model_path}")
    print(f"额外训练步数: {additional_steps:,}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {model_dir}")
    print("=" * 60)

    # 创建环境
    env = DummyVecEnv([make_env(level=1)])
    eval_env = DummyVecEnv([make_env(level=1)])

    # 加载现有模型
    model = DQN.load(model_path, env=env)

    # 回调
    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="dqn_continue"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir + "/best",
            log_path=log_dir + "/eval",
            eval_freq=CALLBACK_CONFIG['eval_freq'],
            n_eval_episodes=10,
        )
    ]

    print("\n开始继续训练...")
    try:
        model.learn(
            total_timesteps=additional_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")

    # 保存
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\n模型已保存到: {final_model_path}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"查看训练曲线: tensorboard --logdir {log_dir}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="继续训练 DQN 模型")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--steps", type=int, default=None, help="额外训练步数")

    args = parser.parse_args()

    continue_training(model_path=args.model, additional_steps=args.steps)
