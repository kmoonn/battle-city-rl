#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战强化学习配置模块
集中管理所有训练参数和环境配置
"""

from datetime import datetime

# ============================================================
# 游戏环境配置
# ============================================================

# 观察空间维度
OBSERVATION_DIM = 727

# 动作空间: 0-无操作, 1-上, 2-右, 3-下, 4-左, 5-开火
NUM_ACTIONS = 6
ACTION_NAMES = ['NOOP', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'FIRE']

# 方向常量
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# 瓦片类型
TILE_EMPTY = 0
TILE_BRICK = 1
TILE_STEEL = 2
TILE_WATER = 3
TILE_GRASS = 4
TILE_FROZE = 5

# 最大步数
MAX_STEPS = 10000

# ============================================================
# 关卡敌人配置 (basic, fast, power, armor)
# ============================================================

LEVELS_ENEMIES = (
    (18, 2, 0, 0), (14, 4, 0, 2), (14, 4, 0, 2), (2, 5, 10, 3), (8, 5, 5, 2),
    (9, 2, 7, 2), (7, 4, 6, 3), (7, 4, 7, 2), (6, 4, 7, 3), (12, 2, 4, 2),
    (5, 5, 4, 6), (0, 6, 8, 6), (0, 8, 8, 4), (0, 4, 10, 6), (0, 2, 10, 8),
    (16, 2, 0, 2), (8, 2, 8, 2), (2, 8, 6, 4), (4, 4, 4, 8), (2, 8, 2, 8),
    (6, 2, 8, 4), (6, 8, 2, 4), (0, 10, 4, 6), (10, 4, 4, 2), (0, 8, 2, 10),
    (4, 6, 4, 6), (2, 8, 2, 8), (15, 2, 2, 1), (0, 4, 10, 6), (4, 8, 4, 4),
    (3, 8, 3, 6), (6, 4, 2, 8), (4, 4, 4, 8), (0, 10, 4, 6), (0, 6, 4, 10)
)

# ============================================================
# PPO 训练配置
# ============================================================

PPO_CONFIG = {
    'total_timesteps': 5_000_000,  # 500万步（关卡难度较高）
    'learning_rate': 3e-4,         # 提高学习率
    'n_steps': 2048,               # 减少步数，更频繁更新
    'batch_size': 64,              # 减小批大小，更稳定的梯度
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,              # 降低探索系数，更专注利用
}

# ============================================================
# DQN 训练配置
# ============================================================

DQN_CONFIG = {
    'total_timesteps': 5_000_000,  # 500万步（关卡难度较高）
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 64,
    'gamma': 0.99,
    'train_freq': 4,
    'gradient_steps': 1,
    'target_update_interval': 1000,
    'exploration_fraction': 0.2,   # 减少探索比例，更快利用
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
}

# ============================================================
# 奖励配置
# ============================================================

REWARD_CONFIG = {
    'score_reward_factor': 1.0 / 5.0,    # 大幅增大得分奖励
    'death_penalty': -1.0,                # 减小死亡惩罚（鼓励冒险）
    'kill_reward': 20.0,                  # 大幅增大击杀奖励
    'time_penalty': -0.05,                # 时间惩罚
    'level_complete_reward': 500.0,       # 大幅增大通关奖励
    'game_over_penalty': -30.0,           # 游戏结束惩罚
    'action_reward': 0.1,                 # 行动奖励
    'no_action_penalty': -0.3,            # 不动惩罚
    'fire_reward': 0.5,                   # 新增：开火奖励（鼓励开火）
}

# ============================================================
# 回调配置
# ============================================================

CALLBACK_CONFIG = {
    'checkpoint_freq': 50000,   # 检查点保存频率
    'eval_freq': 50000,         # 评估频率（增大以减少评估开销）
    'n_eval_episodes': 3,       # 评估回合数（减少以加速）
}

# ============================================================
# 工具函数
# ============================================================

def get_log_dir(algorithm='ppo'):
    """获取日志目录路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./logs/battle_city_{algorithm}_{timestamp}"


def get_model_dir(algorithm='ppo'):
    """获取模型目录路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./models/battle_city_{algorithm}_{timestamp}"


def get_level_enemies(level_nr):
    """获取指定关卡的敌人配置"""
    if level_nr <= 35:
        return LEVELS_ENEMIES[level_nr - 1]
    return LEVELS_ENEMIES[34]
