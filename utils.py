#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战强化学习工具模块
提供共享的工具函数和回调设置
"""

import os
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CALLBACK_CONFIG


def make_env(env_class, level=1, render_mode=None):
    """
    创建环境的工厂函数

    Args:
        env_class: 环境类
        level: 关卡编号
        render_mode: 渲染模式

    Returns:
        环境初始化函数
    """
    def _init():
        env = env_class(render_mode=render_mode, level=level)
        return Monitor(env)
    return _init


def create_vec_env(env_class, level=1, n_envs=1, render_mode=None):
    """
    创建向量化环境

    Args:
        env_class: 环境类
        level: 关卡编号
        n_envs: 环境数量
        render_mode: 渲染模式

    Returns:
        向量化环境
    """
    env_fns = [make_env(env_class, level, render_mode) for _ in range(n_envs)]
    return DummyVecEnv(env_fns)


def create_callbacks(model_dir, log_dir, config=None):
    """
    创建训练回调

    Args:
        model_dir: 模型保存目录
        log_dir: 日志目录
        config: 回调配置，默认使用 CALLBACK_CONFIG

    Returns:
        回调列表
    """
    if config is None:
        config = CALLBACK_CONFIG

    callbacks = []

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'],
        save_path=model_dir,
        name_prefix="battle_city",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # 评估回调 - 注意：调用者需要自己创建评估环境
    # 这里只创建检查点回调
    eval_callback = None  # 需要在调用时传入 eval_env

    return callbacks


def setup_directories(*dirs):
    """
    创建目录（如果不存在）

    Args:
        *dirs: 目录路径列表
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_training_info(algorithm, config, log_dir, model_dir):
    """
    打印训练信息

    Args:
        algorithm: 算法名称
        config: 训练配置
        log_dir: 日志目录
        model_dir: 模型目录
    """
    print("=" * 60)
    print(f"坦克大战 {algorithm.upper()} 训练")
    print("=" * 60)
    print(f"总训练步数: {config['total_timesteps']:,}")
    print(f"学习率: {config['learning_rate']}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {model_dir}")
    print("=" * 60)


def print_episode_result(episode, total_episodes, info, episode_reward, step):
    """
    打印回合结果

    Args:
        episode: 当前回合
        total_episodes: 总回合数
        info: 环境信息字典
        episode_reward: 回合奖励
        step: 步数
    """
    result = "通关! 🎉" if info.get('level_complete', False) else "失败"
    print(f"\n回合 {episode}/{total_episodes} 结果: {result}")
    print(f"  总步数: {step}")
    print(f"  总奖励: {episode_reward:.2f}")
    print(f"  得分: {info.get('score', 0)}")
    print(f"  击杀敌人: {info.get('total_kills', 0)}")
    print(f"  剩余生命: {info.get('lives', 0)}")


def find_latest_model(base_dir="./models", pattern="battle_city_*/final_model.zip"):
    """
    查找最新的模型文件

    Args:
        base_dir: 模型基础目录
        pattern: 模型文件模式

    Returns:
        最新模型路径，未找到返回 None
    """
    import glob
    search_path = os.path.join(base_dir, pattern)
    model_files = sorted(glob.glob(search_path))
    return model_files[-1] if model_files else None
