#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战 PPO 训练脚本
"""

import os
import sys

# 设置无头模式
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import BattleCityEnv
from config import PPO_CONFIG, CALLBACK_CONFIG, get_log_dir, get_model_dir
from utils import setup_directories, print_training_info, print_episode_result, find_latest_model


def make_env(level=1):
    """创建环境的工厂函数"""
    def _init():
        env = BattleCityEnv(render_mode=None, level=level)
        return Monitor(env)
    return _init


def train():
    """主训练函数"""
    # 获取配置
    config = PPO_CONFIG

    # 创建保存目录
    log_dir = get_log_dir('ppo')
    model_dir = get_model_dir('ppo')
    setup_directories(log_dir, model_dir)

    # 打印训练信息
    print_training_info('PPO', config, log_dir, model_dir)

    # 创建环境
    print("\n创建训练环境...")
    env = DummyVecEnv([make_env(level=1)])

    print("创建评估环境...")
    eval_env = DummyVecEnv([make_env(level=1)])

    # 创建模型
    print("\n初始化 PPO 模型...")
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
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    # 创建回调
    callbacks = [
        CheckpointCallback(
            save_freq=CALLBACK_CONFIG['checkpoint_freq'],
            save_path=model_dir,
            name_prefix="battle_city",
            save_replay_buffer=False,
            save_vecnormalize=True
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir + "/best",
            log_path=log_dir + "/eval",
            eval_freq=CALLBACK_CONFIG['eval_freq'],
            n_eval_episodes=CALLBACK_CONFIG['n_eval_episodes'],
            deterministic=True
        )
    ]

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

    # 关闭环境
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"\n使用以下命令查看训练曲线:")
    print(f"  tensorboard --logdir {log_dir}")

    return model


def test_model(model_path=None):
    """测试训练好的模型"""
    print("\n" + "=" * 60)
    print("测试模型")
    print("=" * 60)

    # 创建环境
    env = BattleCityEnv(render_mode=None, level=1)

    # 加载模型或使用新模型
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("使用随机策略测试...")
        model = None

    # 测试回合
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

            # 每 100 步打印一次状态
            if step % 100 == 0:
                print(f"  步 {step}: 奖励={episode_reward:.2f}, "
                      f"得分={info['score']}, 生命={info['lives']}, "
                      f"剩余敌人={info['enemies_remaining']}")

        total_rewards.append(episode_reward)
        total_scores.append(info['score'])

        # 打印回合结果
        print_episode_result(ep + 1, n_episodes, info, episode_reward, step)

    env.close()

    # 打印统计
    print("\n" + "-" * 40)
    print(f"平均奖励: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"平均得分: {sum(total_scores) / len(total_scores):.0f}")

    return total_rewards, total_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="坦克大战强化学习训练")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--test", action="store_true", help="测试模型")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--latest", action="store_true", help="使用最新模型")

    args = parser.parse_args()

    # 查找最新模型
    if args.latest:
        latest = find_latest_model()
        if latest:
            args.model = latest
            print(f"使用最新模型: {args.model}")
        else:
            print("未找到已训练的模型")

    if args.train:
        train()
    elif args.test:
        test_model(args.model)
    else:
        # 默认：训练然后测试
        print("开始训练和测试流程...\n")
        model = train()
        print("\n训练完成，开始测试...")
        test_model()
