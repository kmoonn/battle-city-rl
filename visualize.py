#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战可视化测试脚本
显示游戏画面并测试训练好的模型
"""

import os
import sys
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ACTION_NAMES
from utils import print_episode_result, find_latest_model


def test_with_visualization(model_path=None, n_episodes=3, speed=1.0):
    """
    可视化测试模型

    Args:
        model_path: 模型路径，None 则使用随机策略
        n_episodes: 测试回合数
        speed: 游戏速度 (1.0 = 正常, 0.5 = 慢速, 2.0 = 快速)
    """
    import pygame
    import numpy as np
    from stable_baselines3 import PPO, DQN

    # 导入环境
    from battle_city_env import BattleCityEnv

    # 创建带渲染的环境
    print("创建可视化环境...")
    env = BattleCityEnv(render_mode="human", level=1)

    # 加载模型
    model = None
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        # 根据文件路径判断模型类型
        if "level_" in model_path or "dqn" in model_path.lower():
            print("使用 DQN 加载器")
            model = DQN.load(model_path, env=env)
        else:
            print("使用 PPO 加载器")
            model = PPO.load(model_path, env=env)
    else:
        print("使用随机策略测试...")

    print("\n" + "=" * 60)
    print("可视化测试")
    print("=" * 60)
    print(f"回合数: {n_episodes}")
    print(f"游戏速度: {speed}x")
    print("按 ESC 退出, 按 P 暂停")
    print("=" * 60 + "\n")

    # 测试回合
    total_rewards = []
    total_scores = []
    total_kills = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        paused = False

        print(f"回合 {ep + 1}/{n_episodes} 开始...")

        while not done:
            # 处理 pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n用户退出")
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\n用户退出")
                        env.close()
                        return
                    elif event.key == pygame.K_p:
                        paused = not paused
                        if paused:
                            print("游戏暂停，按 P 继续")
                        else:
                            print("游戏继续")

            if paused:
                time.sleep(0.1)
                continue

            # 获取动作
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            # 渲染
            env.render()

            # 控制游戏速度
            time.sleep(0.02 / speed)  # 基础帧率约 50 FPS

        total_rewards.append(episode_reward)
        total_scores.append(info['score'])
        total_kills.append(info.get('total_kills', 0))

        # 打印回合结果
        print_episode_result(ep + 1, n_episodes, info, episode_reward, step)
        print()

        # 回合间暂停
        if ep < n_episodes - 1:
            print("下一回合将在 2 秒后开始...")
            time.sleep(2)

    env.close()

    # 打印统计
    print("\n" + "=" * 60)
    print("测试统计")
    print("=" * 60)
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"平均得分: {np.mean(total_scores):.0f}")
    print(f"平均击杀: {np.mean(total_kills):.1f}")
    print(f"最高得分: {max(total_scores)}")
    print(f"最高击杀: {max(total_kills)}")
    print("=" * 60)


def play_manually():
    """手动玩游戏模式"""
    import pygame

    print("\n" + "=" * 60)
    print("手动游戏模式")
    print("=" * 60)
    print("控制:")
    print("  方向键: 移动")
    print("  空格: 开火")
    print("  ESC: 退出")
    print("=" * 60 + "\n")

    # 运行原始游戏
    os.system(f"./battle/bin/python tanks.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="坦克大战可视化测试")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--episodes", type=int, default=3, help="测试回合数")
    parser.add_argument("--speed", type=float, default=1.0, help="游戏速度")
    parser.add_argument("--play", action="store_true", help="手动游戏模式")
    parser.add_argument("--latest", action="store_true", help="使用最新模型")
    parser.add_argument("--dqn", action="store_true", help="使用最新 DQN 模型")

    args = parser.parse_args()

    # 查找最新模型
    if args.latest:
        latest = find_latest_model()
        if latest:
            args.model = latest
            print(f"使用最新模型: {args.model}")
        else:
            print("未找到已训练的模型，使用随机策略")

    if args.dqn:
        latest = find_latest_model(pattern="battle_city_dqn_*/final_model.zip")
        if latest:
            args.model = latest
            print(f"使用最新 DQN 模型: {args.model}")
        else:
            print("未找到 DQN 模型，使用随机策略")

    if args.play:
        play_manually()
    else:
        test_with_visualization(
            model_path=args.model,
            n_episodes=args.episodes,
            speed=args.speed
        )
