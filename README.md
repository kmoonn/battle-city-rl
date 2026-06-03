# 🎮 Battle City RL - 坦克大战强化学习

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-DQN-green.svg)](https://github.com/DLR-RM/stable-baselines3)

使用深度强化学习训练 AI 通关经典坦克大战游戏！

![Battle City](https://raw.githubusercontent.com/raitisg/battle-city-tanks/master/images/screens/02.png)

## ✨ 特性

- 🤖 **DQN 强化学习** - 使用 Deep Q-Network 训练智能体
- 🏆 **自动模型保存** - 按关卡保存最佳模型
- 📊 **TensorBoard 可视化** - 实时监控训练过程
- 🎮 **可视化测试** - 支持高速回放观看 AI 表现
- ⚙️ **模块化设计** - 配置与代码分离，易于扩展

## 📦 安装

### 克隆仓库

```bash
git clone https://github.com/kmoonn/battle-city-rl.git
cd battle-city-rl
```

### 创建环境

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 安装依赖

推荐使用 uv

```bash
uv pip install -r requirements.txt
```

依赖列表：
- `numpy>=1.21.0`
- `pygame>=2.0.0`
- `gymnasium>=0.29.0`
- `stable-baselines3>=2.0.0`

## 🚀 快速开始

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
# 测试关卡 1 的模型
python test.py --level 1 --episodes 10

# 可视化测试（显示游戏画面）
python test.py --visual --level 1 --episodes 5 --speed 10

# 测试指定模型
python test.py --algo dqn --model ./models/dqn/level_1/final_model.zip --episodes 5

# 使用 PPO 模型测试
python test.py --ppo --level 1
```

### 手动游戏

```bash
python test.py --play --level 1
```

## 📁 项目结构

```
/
├── tanks.py              # 游戏引擎（原版 Battle City）
├── env.py                # Gymnasium 环境包装器
├── config.py             # 配置模块（超参数、奖励设置）
├── utils.py              # 工具函数
├── train.py              # 统一训练脚本（DQN/PPO）
├── test.py               # 统一测试脚本
├── requirements.txt      # Python 依赖
│
├── models/               # 训练保存的模型
│   ├── dqn/
│   │   └── level_1/
│   │       ├── final_model.zip
│   │       └── checkpoint_10000_steps.zip
│   └── ppo/
│       └── level_1/
│           └── final_model.zip
│
├── logs/                 # TensorBoard 日志（按算法/关卡划分）
├── levels/               # 关卡地图文件
├── images/               # 游戏图片资源
├── sounds/               # 音效文件
└── fonts/                # 字体文件
```

## 🎯 观察空间

观察空间是 **727 维**向量，包含：

| 组件   | 维度  | 说明               |
|------|-----|------------------|
| 玩家状态 | 4   | x, y 坐标, 方向, 生命值 |
| 敌人状态 | 16  | 最多 4 个敌人 × 4 属性  |
| 子弹状态 | 30  | 最多 10 颗子弹 × 3 属性 |
| 地图状态 | 676 | 26×26 网格的瓦片类型    |
| 剩余敌人 | 1   | 剩余敌人数量           |

## 🕹️ 动作空间

**6 个离散动作**：

| 动作 | 说明         |
|----|------------|
| 0  | 无操作 (NOOP) |
| 1  | 向上移动       |
| 2  | 向右移动       |
| 3  | 向下移动       |
| 4  | 向左移动       |
| 5  | 开火         |

## ⚙️ 配置说明

主要配置位于 `config.py`：

### DQN 超参数

```python
DQN_CONFIG = {
    'total_timesteps': 2_000_000,  # 总训练步数
    'learning_rate': 1e-4,         # 学习率
    'buffer_size': 100000,         # 经验回放缓冲区大小
    'learning_starts': 10000,      # 开始学习的步数
    'batch_size': 64,              # 批大小
    'gamma': 0.99,                 # 折扣因子
    'exploration_fraction': 0.3,   # 探索比例
}
```

### 奖励函数

```python
REWARD_CONFIG = {
    'score_reward_factor': 1.0/30.0,   # 得分奖励系数
    'death_penalty': -1.5,              # 死亡惩罚
    'kill_reward': 2.0,                 # 击杀奖励
    'time_penalty': -0.01,              # 时间惩罚
    'level_complete_reward': 50.0,      # 通关奖励
    'game_over_penalty': -5.0,          # 游戏结束惩罚
}
```

## 📊 训练结果

### 关卡 1 性能

| 指标   | 数值        |
|------|-----------|
| 通关率  | **100%**  |
| 平均奖励 | **92.09** |
| 平均得分 | **233**   |
| 最快通关 | **184 步** |

### 训练曲线

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./logs
```

日志会按算法和关卡分目录保存，例如 `./logs/dqn/level_1`。

然后在浏览器打开 `http://localhost:6006`

## 🔧 高级用法

### 继续训练

```bash
# 自动从当前算法的已有模型继续训练
python train.py --algo dqn --level 1

# 指定训练步数
python train.py --algo dqn --level 1 --steps 1000000
```

### 不同算法

```bash
# DQN 算法（默认）
python train.py --algo dqn --level 1

# PPO 算法
python train.py --algo ppo --level 1
```

### 测试选项

测试和自动发现会优先使用当前算法目录下的模型，例如 DQN 使用 `./models/dqn/level_1/...`，PPO 使用 `./models/ppo/level_1/...`。

```bash
# 无头测试
python test.py --algo dqn --level 1 --episodes 10

# 可视化测试
python test.py --visual --algo dqn --level 1 --speed 10

# 手动游戏
python test.py --play --level 1
```

### 自定义奖励

修改 `config.py` 中的 `REWARD_CONFIG` 来调整奖励函数：

```python
# 更鼓励攻击性行为
REWARD_CONFIG['kill_reward'] = 3.0        # 增加击杀奖励
REWARD_CONFIG['time_penalty'] = -0.02     # 增加时间压力
```

## 🎓 技术细节

### 环境设计

- 继承 `gymnasium.Env`，符合 Gym API 规范
- 支持无头模式（`SDL_VIDEODRIVER=dummy`）用于快速训练
- 每步对应约 20ms 游戏时间

### 网络结构

使用 Stable-Baselines3 的默认 MlpPolicy：
- 2 层全连接网络
- 每层 64 个神经元
- ReLU 激活函数

### 探索策略

- 使用 ε-greedy 策略
- 初始 ε = 1.0，最终 ε = 0.05
- 探索比例占总训练步数的 30%

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


## 🙏 参考

- 游戏引擎基于 [battle-city-tanks](https://github.com/raitisg/battle-city-tanks)
- 强化学习框架使用 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- 环境接口遵循 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 标准

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📮 联系方式

如有问题或建议，欢迎：
- 提交 [Issue](https://github.com/kmoonn/battle-city-rl/issues)
- 发起 [Discussion](https://github.com/kmoonn/battle-city-rl/discussions)

---

⭐ 如果这个项目对你有帮助，请给一个 Star！
