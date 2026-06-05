## 🚀 关卡一快速开始

### 在完成基础环境搭建后，你可以直接运行看到level 1的最佳训练模型结果
**使用PPO模型进行训练**
```bash
python test.py --model ./models/level_2/best_model_ppo.zip --episodes 5 --speed 30 --visual

```

### 所使用的dpn超参数和奖励函数配置为

```python
# PPO 训练配置
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
```

```python
# 奖励配置
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
```


### 训练模型

```bash
# 训练关卡 1（默认 DQN）
python train.py --level 1

# 使用 PPO 算法
python train.py --algo ppo --level 1

# 指定训练步数
python train.py --level 1 --steps 5000000

# 从头开始训练（不加载已有模型）
python train.py --level 1 --new
```

### 测试模型

```bash
# 测试关卡 1 的 DQN 模型
python test.py --algo dqn --level 1 --episodes 10

# 测试关卡 1 的 PPO 模型
python test.py --algo ppo --level 1 --episodes 10

# 可视化测试（显示游戏画面）
python test.py --visual --algo ppo --level 1 --speed 3

# 测试指定模型
python test.py --model ./models/level_1/best_model_dqn.zip --episodes 5
```