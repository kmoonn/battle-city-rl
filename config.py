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
    'learning_rate': 3e-4,         # 学习率
    'n_steps': 2048,               # 每轮采集步数
    'batch_size': 64,              # 批大小
    'n_epochs': 10,
    'gamma': 0.995,                # 折扣因子，重视长期回报（通关）
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.02,              # 适中探索系数，平衡探索与利用
    'vf_coef': 0.5,                # 价值函数系数
    'max_grad_norm': 0.5,          # 梯度裁剪
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
    # --- 核心目标奖励 ---
    'kill_reward': 30.0,                  # 击杀奖励（只有开火才能击杀，核心驱动）
    'level_complete_reward': 500.0,       # 通关奖励
    'game_over_penalty': -30.0,           # 游戏结束惩罚
    
    # --- 每步基础奖励 ---
    'time_penalty': -0.05,                # 轻微时间惩罚（鼓励效率）
    'death_penalty': -1.0,                # 死亡惩罚（较轻，不让agent因怕死而不战）
    
    # --- 动作奖励（简单清晰）---
    'fire_reward': 0.3,                   # 开火奖励：每步开火都能获得，确保agent愿意射击
    'move_reward': 0.0,                   # 移动不给奖励（消除"只移动"的诱惑）
    'noop_penalty': -0.1,                 # 不动轻微惩罚
}

# ============================================================
# 回调配置
# ============================================================

CALLBACK_CONFIG = {
    'checkpoint_freq': 50000,   # 检查点保存频率
    'eval_freq': 25000,         # 评估频率（更频繁评估以找到最佳模型）
    'n_eval_episodes': 5,       # 评估回合数（更多回合得到更可靠评估）
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
