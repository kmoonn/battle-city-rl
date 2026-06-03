#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战统一训练脚本
支持 DQN / PPO 算法，支持按关卡训练
"""

import argparse
import os
import sys

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

os.environ['SDL_VIDEODRIVER'] = 'dummy'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CALLBACK_CONFIG, DQN_CONFIG, PPO_CONFIG
from env import BattleCityEnv
from test import test_headless
from utils import (
    create_vec_env,
    find_level_model,
    get_level_log_dir,
    get_level_model_dir,
    load_model,
    normalize_algorithm,
    setup_directories,
)


# ============================================================
# 训练函数
# ============================================================


def train(algorithm='dqn', level=1, total_timesteps=None, continue_from_best=True):
    """
    统一训练函数

    Args:
        algorithm: 算法类型 'dqn' 或 'ppo'
        level: 关卡编号
        total_timesteps: 总训练步数，None 使用默认值
        continue_from_best: 是否从已有模型继续训练
    """
    algorithm = normalize_algorithm(algorithm)

    if algorithm == 'ppo':
        config = PPO_CONFIG
        model_class = PPO
    else:
        config = DQN_CONFIG
        model_class = DQN

    if total_timesteps is None:
        total_timesteps = config['total_timesteps']

    model_dir = get_level_model_dir(level=level, algorithm=algorithm)
    log_dir = get_level_log_dir(level=level, algorithm=algorithm)
    setup_directories(model_dir, log_dir)

    print("=" * 60)
    print(f"坦克大战 {algorithm.upper()} 训练 - 关卡 {level}")
    print("=" * 60)
    print(f"总训练步数: {total_timesteps:,}")
    print(f"模型目录: {model_dir}")
    print(f"日志目录: {log_dir}")

    print("\n创建训练环境...")
    env = create_vec_env(BattleCityEnv, level=level, n_envs=1, render_mode=None)

    model = None
    if continue_from_best:
        resume_path = find_level_model(level=level, algorithm=algorithm)
        if resume_path:
            print(f"从已有模型继续: {resume_path}")
            try:
                model = load_model(resume_path, env=env, algorithm=algorithm)
            except Exception as exc:
                env.close()
                raise RuntimeError(
                    f"加载已有 {algorithm.upper()} 模型失败: {resume_path}\n"
                    f"请检查模型文件是否与算法匹配，或使用 --new 从头开始训练。\n"
                    f"原始错误: {exc}"
                ) from exc
        else:
            print("未找到已有模型，从头开始训练")

    if model is None:
        print(f"创建新 {algorithm.upper()} 模型...")

        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        )

        if algorithm == 'ppo':
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gamma=config['gamma'],
                gae_lambda=config['gae_lambda'],
                clip_range=config['clip_range'],
                ent_coef=config['ent_coef'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
            )
        else:
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                buffer_size=config['buffer_size'],
                learning_starts=config['learning_starts'],
                batch_size=config['batch_size'],
                gamma=config['gamma'],
                train_freq=config['train_freq'],
                gradient_steps=config['gradient_steps'],
                target_update_interval=config['target_update_interval'],
                exploration_fraction=config['exploration_fraction'],
                exploration_initial_eps=config['exploration_initial_eps'],
                exploration_final_eps=config['exploration_final_eps'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
            )

    # 创建评估环境用于保存最佳模型
    eval_env = Monitor(BattleCityEnv(render_mode=None, level=level))

    # 使用 EvalCallback 自动保存最佳模型
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=CALLBACK_CONFIG['eval_freq'],
            n_eval_episodes=CALLBACK_CONFIG['n_eval_episodes'],
            deterministic=True,
            render=False,
        )
    ]

    print("\n开始训练...")
    print(f"每 {CALLBACK_CONFIG['eval_freq']:,} 步评估一次，自动保存最佳模型")
    print("-" * 60)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")

    # 重命名最佳模型文件，添加算法后缀
    old_best_path = os.path.join(model_dir, "best_model.zip")
    new_best_path = os.path.join(model_dir, f"best_model_{algorithm}.zip")
    if os.path.exists(old_best_path):
        # 删除旧的带后缀文件（如果存在）
        if os.path.exists(new_best_path):
            os.remove(new_best_path)
        os.rename(old_best_path, new_best_path)
        print(f"\n最佳模型已保存: {new_best_path}")
    else:
        print(f"\n警告: 未找到最佳模型文件 {old_best_path}")

    # 清理评估日志
    eval_logs_dir = os.path.join(model_dir, "eval_logs")
    if os.path.exists(eval_logs_dir):
        import shutil
        shutil.rmtree(eval_logs_dir)

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型目录: {model_dir}")
    print(f"最佳模型: {new_best_path}")
    print(f"查看训练曲线: tensorboard --logdir {log_dir}")
    print("=" * 60)

    return new_best_path


# ============================================================
# 主函数
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="坦克大战统一训练脚本")

    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--train", action="store_true", help="训练模式（默认）")

    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="算法类型 (default: dqn)")
    parser.add_argument("--level", type=int, default=1, help="关卡编号")
    parser.add_argument("--steps", type=int, default=None, help="训练步数")

    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--model", type=str, default=None, help="模型路径")

    parser.add_argument("--new", action="store_true", help="从头训练（不加载已有模型）")

    args = parser.parse_args()

    if args.test:
        test_headless(
            model_path=args.model,
            level=args.level,
            n_episodes=args.episodes,
            algorithm=args.algo,
            allow_latest=args.model is None,
        )
    else:
        trained_model_path = train(
            algorithm=args.algo,
            level=args.level,
            total_timesteps=args.steps,
            continue_from_best=not args.new,
        )

        print("\n测试模型...")
        test_headless(
            model_path=trained_model_path,
            level=args.level,
            n_episodes=args.episodes,
            algorithm=args.algo,
        )
