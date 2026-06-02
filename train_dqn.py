#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战 DQN 训练脚本
"""

import os
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battle_city_env import BattleCityEnv
from config import DQN_CONFIG, CALLBACK_CONFIG, get_log_dir, get_model_dir
from utils import setup_directories, print_training_info, print_episode_result, find_latest_model


def make_env(level=1):
    """创建环境的工厂函数"""
    def _init():
        env = BattleCityEnv(render_mode=None, level=level)
        return Monitor(env)
    return _init


def train_dqn():
    """DQN 训练函数"""
    # 获取配置
    config = DQN_CONFIG

    # 创建保存目录
    log_dir = get_log_dir('dqn')
    model_dir = get_model_dir('dqn')
    setup_directories(log_dir, model_dir)

    # 打印训练信息
    print_training_info('DQN', config, log_dir, model_dir)

    # 创建环境
    print("\n创建训练环境...")
    env = DummyVecEnv([make_env(level=1)])

    print("创建评估环境...")
    eval_env = DummyVecEnv([make_env(level=1)])

    # 创建 DQN 模型
    print("\n初始化 DQN 模型...")
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
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 创建回调
    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="dqn"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir + "/best",
            log_path=log_dir + "/eval",
            eval_freq=CALLBACK_CONFIG['eval_freq'],
            n_eval_episodes=CALLBACK_CONFIG['n_eval_episodes'],
        )
    ]

    # 训练
    print("\n开始训练...")
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
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


def test_dqn(model_path=None):
    """测试 DQN 模型"""
    print("\n" + "=" * 60)
    print("测试 DQN 模型")
    print("=" * 60)

    env = BattleCityEnv(render_mode=None, level=1)

    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = DQN.load(model_path, env=env)
    else:
        # 尝试查找最新模型
        latest = find_latest_model(pattern="battle_city_dqn_*/final_model.zip")
        if latest:
            print(f"加载最新模型: {latest}")
            model = DQN.load(latest, env=env)
        else:
            print("使用随机策略测试...")
            model = None

    # 测试
    n_episodes = 3
    total_rewards = []
    total_scores = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0

        print(f"\n回合 {ep + 1}/{n_episodes}")

        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if step % 100 == 0:
                print(f"  步 {step}: 奖励={episode_reward:.2f}, "
                      f"得分={info['score']}, 生命={info['lives']}")

        total_rewards.append(episode_reward)
        total_scores.append(info['score'])
        print_episode_result(ep + 1, n_episodes, info, episode_reward, step)

    env.close()

    print("\n" + "-" * 40)
    print(f"平均奖励: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"平均得分: {sum(total_scores) / len(total_scores):.0f}")

    return total_rewards, total_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="坦克大战 DQN 训练")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--test", action="store_true", help="测试模型")
    parser.add_argument("--model", type=str, default=None, help="模型路径")

    args = parser.parse_args()

    if args.train:
        train_dqn()
    elif args.test:
        test_dqn(args.model)
    else:
        # 默认：训练
        train_dqn()
