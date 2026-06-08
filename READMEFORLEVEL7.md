## 🚀 关卡七快速开始

### 在完成基础环境搭建后，你可以直接运行看到level 7的最佳训练模型结果
**使用DPN模型进行训练**
```bash
python test.py --algo dqn --model ./models/level_7/best_model_dqn.zip --level 7  --episodes 10 --speed 30 --visual

```

### 所使用的dpn超参数和奖励函数配置为

```python
# DQN 训练配置
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
```

```python
# 奖励配置
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