#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战强化学习工具模块
提供共享的工具函数和路径辅助
"""

import glob
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
# 环境辅助
# ============================================================


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


# ============================================================
# 模型与路径辅助
# ============================================================


def normalize_algorithm(algorithm='dqn'):
    """规范化算法名称"""
    return 'ppo' if str(algorithm).lower() == 'ppo' else 'dqn'



def get_model_class(algorithm='dqn'):
    """根据算法名称返回对应的 SB3 模型类"""
    from stable_baselines3 import DQN, PPO

    return PPO if normalize_algorithm(algorithm) == 'ppo' else DQN



def get_level_model_dir(level=1, algorithm='dqn', base_dir="./models"):
    """获取关卡模型目录（一个关卡一个文件夹）"""
    return os.path.join(base_dir, f"level_{level}")



def get_level_log_dir(level=1, algorithm='dqn', base_dir="./logs"):
    """获取按算法和关卡划分的日志目录"""
    algorithm = normalize_algorithm(algorithm)
    return os.path.join(base_dir, algorithm, f"level_{level}")



def strip_zip_suffix(model_path):
    """Stable-Baselines3 支持不带 .zip 的保存名前缀，这里统一处理加载路径"""
    if model_path.endswith(".zip"):
        return model_path[:-4]
    return model_path



def load_model(model_path, env, algorithm='dqn'):
    """按算法加载模型"""
    model_class = get_model_class(algorithm)
    return model_class.load(strip_zip_suffix(model_path), env=env)



def _latest_file(paths):
    """按修改时间返回最新文件"""
    existing_paths = [path for path in paths if os.path.exists(path)]
    if not existing_paths:
        return None
    return max(existing_paths, key=os.path.getmtime)



def _latest_pattern(pattern):
    """返回通配匹配到的最新文件"""
    return _latest_file(glob.glob(pattern))



def _find_legacy_level_model(level=1, base_dir="./models"):
    """查找旧版按关卡目录中的模型（仅保留可明确识别为旧 DQN 流程的 best_model）"""
    legacy_dir = os.path.join(base_dir, f"level_{level}")
    # 优先查找新命名格式
    best_model_path = os.path.join(legacy_dir, "best_model_dqn.zip")
    if os.path.exists(best_model_path):
        return best_model_path
    # 兼容旧格式
    best_model_path = os.path.join(legacy_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        return best_model_path
    return None



def find_best_model(base_dir="./models", algorithm='dqn'):
    """
    查找最佳模型文件

    Args:
        base_dir: 模型基础目录
        algorithm: 算法类型

    Returns:
        最佳模型路径，未找到返回 None
    """
    algorithm = normalize_algorithm(algorithm)

    # 查找新格式: models/level_X/best_model_ppo.zip 或 best_model_dqn.zip
    candidates = [
        _latest_pattern(os.path.join(base_dir, "level_*", f"best_model_{algorithm}.zip")),
    ]

    return _latest_file([path for path in candidates if path])



def find_level_model(level=1, algorithm='dqn', base_dir="./models"):
    """
    查找指定关卡的最佳可用模型

    查找顺序：
    1. models/level_X/best_model_ppo.zip 或 best_model_dqn.zip
    2. 兼容旧格式: models/level_X/best_model.zip

    Args:
        level: 关卡编号
        algorithm: 算法类型
        base_dir: 模型基础目录

    Returns:
        模型路径，未找到返回 None
    """
    algorithm = normalize_algorithm(algorithm)
    model_dir = get_level_model_dir(level, algorithm, base_dir)

    # 新格式: best_model_ppo.zip 或 best_model_dqn.zip
    best_model_path = os.path.join(model_dir, f"best_model_{algorithm}.zip")
    if os.path.exists(best_model_path):
        return best_model_path

    # 兼容旧格式: best_model.zip
    legacy_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(legacy_path):
        return legacy_path

    return None



def find_latest_model(base_dir="./models", algorithm='dqn'):
    """
    查找最新的模型文件

    Args:
        base_dir: 模型基础目录
        algorithm: 算法类型

    Returns:
        最新模型路径，未找到返回 None
    """
    algorithm = normalize_algorithm(algorithm)

    # 查找新格式
    candidates = [
        _latest_pattern(os.path.join(base_dir, "level_*", f"best_model_{algorithm}.zip")),
    ]

    return _latest_file([path for path in candidates if path])



def resolve_model_path(model_path=None, level=1, algorithm='dqn', base_dir="./models", allow_latest=False):
    """解析模型路径：优先显式路径，否则按关卡查找；可选回退到最新同算法模型"""
    if model_path:
        return model_path

    resolved_path = find_level_model(level=level, algorithm=algorithm, base_dir=base_dir)
    if resolved_path:
        return resolved_path

    if allow_latest:
        return find_latest_model(base_dir=base_dir, algorithm=algorithm)

    return None


# ============================================================
# 通用辅助
# ============================================================


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
