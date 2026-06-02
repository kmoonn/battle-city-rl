#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按关卡训练脚本 - 每个关卡只保存最佳模型
"""

import os
import sys
import glob
import shutil

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battle_city_env import BattleCityEnv
from config import DQN_CONFIG, CALLBACK_CONFIG


class BestModelCallback(BaseCallback):
    """只保存最佳模型的回调"""

    def __init__(self, eval_env, model_dir, eval_freq=10000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.model_dir = model_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -float('inf')
        self.eval_count = 0

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            mean_reward, std_reward = self._evaluate()

            if self.verbose > 0:
                print(f"\n评估 #{self.eval_count}: 平均奖励={mean_reward:.2f} ± {std_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.model_dir, "best_model.zip")
                self.model.save(model_path.replace(".zip", ""))
                if self.verbose > 0:
                    print(f"  新最佳模型! 奖励={mean_reward:.2f} -> 已保存")

            return True
        return True

    def _evaluate(self):
        """评估模型"""
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)

        import numpy as np
        return np.mean(episode_rewards), np.std(episode_rewards)


def make_env(level=1):
    """创建环境"""
    def _init():
        env = BattleCityEnv(render_mode=None, level=level)
        return Monitor(env)
    return _init


def find_best_model(level=1):
    """查找指定关卡的最佳模型"""
    model_dir = f"./models/level_{level}"
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        return best_model_path

    # 也检查旧格式的模型
    old_models = sorted(glob.glob(f"./models/battle_city_dqn*/best/best_model.zip"))
    if old_models:
        return old_models[-1]

    return None


def train_level(level=1, total_timesteps=None, continue_from_best=True):
    """训练指定关卡

    Args:
        level: 关卡编号
        total_timesteps: 总训练步数
        continue_from_best: 是否从最佳模型继续训练
    """
    # 设置训练步数
    if total_timesteps is None:
        total_timesteps = DQN_CONFIG['total_timesteps']

    # 模型目录 - 按关卡组织
    model_dir = f"./models/level_{level}"
    log_dir = f"./logs/level_{level}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"训练关卡 {level}")
    print("=" * 60)
    print(f"总训练步数: {total_timesteps:,}")
    print(f"模型目录: {model_dir}")
    print(f"日志目录: {log_dir}")

    # 创建环境
    env = DummyVecEnv([make_env(level=level)])
    eval_env = BattleCityEnv(render_mode=None, level=level)

    # 检查是否继续训练
    model = None
    if continue_from_best:
        best_model_path = find_best_model(level)
        if best_model_path:
            print(f"从最佳模型继续: {best_model_path}")
            model = DQN.load(best_model_path.replace(".zip", ""), env=env)
        else:
            print("未找到已有模型，从头开始训练")

    if model is None:
        print("创建新 DQN 模型...")
        config = DQN_CONFIG
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
    callback = BestModelCallback(
        eval_env=eval_env,
        model_dir=model_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        verbose=1
    )

    # 训练
    print("\n开始训练...")
    print("-" * 60)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")

    # 保存最终模型
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n最终模型已保存: {final_path}")

    # 清理旧的检查点，只保留最佳模型
    cleanup_checkpoints(model_dir)

    env.close()

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型: {model_dir}/best_model.zip")
    print(f"最佳奖励: {callback.best_mean_reward:.2f}")
    print("=" * 60)

    return model


def cleanup_checkpoints(model_dir):
    """清理检查点，只保留最佳模型"""
    # 删除所有检查点文件
    for pattern in ["battle_city_*.zip", "dqn_*.zip", "rl_model_*.zip"]:
        for f in glob.glob(os.path.join(model_dir, pattern)):
            os.remove(f)
            print(f"已删除检查点: {f}")

    # 删除 best 子目录（如果存在）
    best_dir = os.path.join(model_dir, "best")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
        print(f"已删除目录: {best_dir}")


def test_level(level=1, n_episodes=5):
    """测试指定关卡"""
    print("\n" + "=" * 60)
    print(f"测试关卡 {level}")
    print("=" * 60)

    model_path = find_best_model(level)
    if not model_path:
        print("未找到模型!")
        return

    print(f"加载模型: {model_path}")

    env = BattleCityEnv(render_mode=None, level=level)
    model = DQN.load(model_path.replace(".zip", ""), env=env)

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        level_complete = info.get('level_complete', False)
        results.append({
            'episode': ep + 1,
            'reward': total_reward,
            'steps': step,
            'score': info.get('score', 0),
            'kills': info.get('total_kills', 0),
            'complete': level_complete
        })

        status = "通关 🎉" if level_complete else "失败"
        print(f"回合 {ep+1}: {status} | 步数={step} | 奖励={total_reward:.2f} | 得分={info.get('score', 0)}")

    env.close()

    # 统计
    wins = sum(1 for r in results if r['complete'])
    avg_reward = sum(r['reward'] for r in results) / len(results)

    print("\n" + "-" * 40)
    print(f"通关率: {wins}/{n_episodes} ({100*wins/n_episodes:.0f}%)")
    print(f"平均奖励: {avg_reward:.2f}")
    print("-" * 40)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按关卡训练")
    parser.add_argument("--level", type=int, default=1, help="关卡编号")
    parser.add_argument("--steps", type=int, default=None, help="训练步数")
    parser.add_argument("--test", action="store_true", help="只测试不训练")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--new", action="store_true", help="从头训练（不加载已有模型）")

    args = parser.parse_args()

    if args.test:
        test_level(args.level, args.episodes)
    else:
        train_level(
            level=args.level,
            total_timesteps=args.steps,
            continue_from_best=not args.new
        )
        print("\n测试模型...")
        test_level(args.level, args.episodes)
