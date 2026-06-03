#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战统一测试脚本
支持无头测试和可视化测试
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import BattleCityEnv
from utils import (
    load_model,
    normalize_algorithm,
    print_episode_result,
    resolve_model_path,
)


# ============================================================
# 公共辅助
# ============================================================


def _resolve_loaded_model(env, model_path=None, level=1, algorithm='dqn', allow_latest=False):
    """解析并加载模型；未找到时返回 (None, None)"""
    algorithm = normalize_algorithm(algorithm)
    resolved_path = resolve_model_path(
        model_path=model_path,
        level=level,
        algorithm=algorithm,
        allow_latest=allow_latest,
    )

    if not resolved_path or not os.path.exists(resolved_path):
        return None, None

    try:
        model = load_model(resolved_path, env=env, algorithm=algorithm)
    except Exception as exc:
        raise RuntimeError(
            f"加载 {algorithm.upper()} 模型失败: {resolved_path}\n"
            f"请检查 --algo/--ppo/--dqn 与模型是否匹配。\n"
            f"原始错误: {exc}"
        ) from exc

    return model, resolved_path



def _run_test_episodes(env, model, n_episodes, visual=False, speed=1.0):
    """执行测试回合"""
    results = []

    if visual:
        import pygame

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        paused = False

        if visual:
            print(f"回合 {ep + 1}/{n_episodes} 开始...")
        else:
            print(f"\n回合 {ep + 1}/{n_episodes}")

        while not done:
            if visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n用户退出")
                        env.close()
                        return results
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\n用户退出")
                            env.close()
                            return results
                        elif event.key == pygame.K_p:
                            paused = not paused
                            print("游戏暂停" if paused else "游戏继续")

                if paused:
                    time.sleep(0.1)
                    continue

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if visual:
                env.render()
                time.sleep(0.02 / speed)

            if step % 100 == 0:
                print(f"  步 {step}: 奖励={total_reward:.2f}, 得分={info['score']}, 生命={info['lives']}")

        results.append({
            'reward': total_reward,
            'steps': step,
            'score': info.get('score', 0),
            'kills': info.get('total_kills', 0),
            'complete': info.get('level_complete', False),
        })

        print_episode_result(ep + 1, n_episodes, info, total_reward, step)

        if visual:
            print()
            if ep < n_episodes - 1:
                print("下一回合将在 2 秒后开始...")
                time.sleep(2)

    return results



def _print_statistics(results, n_episodes):
    """打印统计信息"""
    if not results:
        print("\n未产生有效测试结果。")
        return

    wins = sum(1 for r in results if r['complete'])
    avg_reward = sum(r['reward'] for r in results) / len(results)
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_kills = sum(r['kills'] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("测试统计")
    print("=" * 60)
    print(f"通关率: {wins}/{n_episodes} ({100*wins/n_episodes:.0f}%)")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均得分: {avg_score:.0f}")
    print(f"平均击杀: {avg_kills:.1f}")
    print(f"最高得分: {max(r['score'] for r in results)}")
    print(f"最高击杀: {max(r['kills'] for r in results)}")
    print("=" * 60)


# ============================================================
# 测试模式
# ============================================================


def test_headless(model_path=None, level=1, n_episodes=5, algorithm='dqn', allow_latest=False):
    """
    无头模式测试

    Args:
        model_path: 模型路径
        level: 关卡编号
        n_episodes: 测试回合数
        algorithm: 算法类型
        allow_latest: 未找到当前关卡模型时，是否回退到同算法的最新模型
    """
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    algorithm = normalize_algorithm(algorithm)

    print("\n" + "=" * 60)
    print(f"无头测试 - {algorithm.upper()} - 关卡 {level}")
    print("=" * 60)

    env = BattleCityEnv(render_mode=None, level=level)

    model, resolved_path = _resolve_loaded_model(
        env,
        model_path=model_path,
        level=level,
        algorithm=algorithm,
        allow_latest=allow_latest,
    )

    if resolved_path:
        print(f"加载模型: {resolved_path}")
    else:
        print("未找到模型，使用随机策略测试...")

    results = _run_test_episodes(env, model, n_episodes, visual=False)
    env.close()
    _print_statistics(results, n_episodes)
    return results



def test_visual(model_path=None, level=1, n_episodes=3, speed=1.0, algorithm='dqn', allow_latest=False):
    """
    可视化测试

    Args:
        model_path: 模型路径
        level: 关卡编号
        n_episodes: 测试回合数
        speed: 游戏速度
        algorithm: 算法类型
        allow_latest: 未找到当前关卡模型时，是否回退到同算法的最新模型
    """
    algorithm = normalize_algorithm(algorithm)

    print("\n" + "=" * 60)
    print(f"可视化测试 - {algorithm.upper()} - 关卡 {level}")
    print("=" * 60)
    print(f"回合数: {n_episodes}")
    print(f"游戏速度: {speed}x")
    print("按 ESC 退出, 按 P 暂停")
    print("=" * 60 + "\n")

    env = BattleCityEnv(render_mode="human", level=level)

    model, resolved_path = _resolve_loaded_model(
        env,
        model_path=model_path,
        level=level,
        algorithm=algorithm,
        allow_latest=allow_latest,
    )

    if resolved_path:
        print(f"加载模型: {resolved_path}")
    else:
        print("未找到模型，使用随机策略测试...")

    results = _run_test_episodes(env, model, n_episodes, visual=True, speed=speed)
    env.close()
    _print_statistics(results, n_episodes)
    return results


# ============================================================
# 手动游戏
# ============================================================


def play_manually(level=None):
    """手动玩游戏模式

    Args:
        level: 关卡编号，None 则进入菜单选择
    """
    print("\n" + "=" * 60)
    print("手动游戏模式")
    print("=" * 60)
    print("控制:")
    print("  方向键: 移动")
    print("  空格: 开火")
    print("  ESC: 退出")
    if level:
        print(f"  关卡: {level}")
    print("=" * 60 + "\n")

    import tanks

    argv = []
    if level:
        argv.extend(["--level", str(level)])
    tanks.main(argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="坦克大战统一测试脚本")

    parser.add_argument("--visual", action="store_true", help="可视化测试模式")
    parser.add_argument("--play", action="store_true", help="手动游戏模式")

    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--level", type=int, default=1, help="关卡编号")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--speed", type=float, default=1.0, help="游戏速度（可视化模式）")

    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="算法类型")
    parser.add_argument("--dqn", action="store_true", help="使用 DQN 模型")
    parser.add_argument("--ppo", action="store_true", help="使用 PPO 模型")

    args = parser.parse_args()

    algorithm = normalize_algorithm(args.algo)
    if args.ppo:
        algorithm = 'ppo'
    elif args.dqn:
        algorithm = 'dqn'

    if args.play:
        play_manually(level=args.level)
    elif args.visual:
        test_visual(
            model_path=args.model,
            level=args.level,
            n_episodes=args.episodes,
            speed=args.speed,
            algorithm=algorithm,
            allow_latest=args.model is None,
        )
    else:
        test_headless(
            model_path=args.model,
            level=args.level,
            n_episodes=args.episodes,
            algorithm=algorithm,
        )
