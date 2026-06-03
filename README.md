# 🎮 Battle City RL - 坦克大战强化学习

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-DQN%2FPPO-green.svg)](https://github.com/DLR-RM/stable-baselines3)

使用深度强化学习训练 AI 通关经典坦克大战游戏！

![Battle City](https://raw.githubusercontent.com/raitisg/battle-city-tanks/master/images/screens/02.png)

## ✨ 特性

- 🤖 **DQN & PPO 算法** - 支持两种主流强化学习算法
- 🏆 **自动保存最佳模型** - 按关卡保存，支持多种算法并存
- 📊 **TensorBoard 可视化** - 实时监控训练过程
- 🎮 **可视化测试** - 支持高速回放观看 AI 表现
- ⚙️ **模块化设计** - 配置与代码分离，易于扩展
- 🎯 **课程学习** - 支持从简单关卡迁移学习

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

```bash
uv pip install -r requirements.txt
```

依赖列表：
- `numpy>=1.21.0`
- `pygame>=2.0.0`
- `gymnasium>=0.29.0`
- `stable-baselines3>=2.0.0`
- `tensorboard>=2.0.0`
- `tqdm>=4.0.0`
- `rich>=13.0.0`

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
# 测试关卡 1 的 DQN 模型
python test.py --algo dqn --level 1 --episodes 10

# 测试关卡 1 的 PPO 模型
python test.py --algo ppo --level 1 --episodes 10

# 可视化测试（显示游戏画面）
python test.py --visual --algo ppo --level 1 --speed 3

# 测试指定模型
python test.py --model ./models/level_1/best_model_dqn.zip --episodes 5
```

### 手动游戏

```bash
# 进入菜单选择关卡
python test.py --play

# 直接玩关卡 2
python test.py --play --level 2
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
│   ├── level_1/
│   │   ├── best_model_dqn.zip
│   │   └── best_model_ppo.zip
│   └── level_2/
│       ├── best_model_dqn.zip
│       └── best_model_ppo.zip
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

### PPO 超参数

```python
PPO_CONFIG = {
    'total_timesteps': 5_000_000,  # 总训练步数
    'learning_rate': 3e-4,         # 学习率
    'n_steps': 2048,               # 每次更新的步数
    'batch_size': 64,              # 批大小
    'n_epochs': 10,                # 训练轮数
    'gamma': 0.99,                 # 折扣因子
    'clip_range': 0.2,             # PPO 裁剪范围
}
```

### DQN 超参数

```python
DQN_CONFIG = {
    'total_timesteps': 5_000_000,  # 总训练步数
    'learning_rate': 1e-4,         # 学习率
    'buffer_size': 100000,         # 经验回放缓冲区大小
    'learning_starts': 10000,      # 开始学习的步数
    'batch_size': 64,              # 批大小
    'gamma': 0.99,                 # 折扣因子
    'exploration_fraction': 0.2,   # 探索比例
}
```

### 奖励函数

```python
REWARD_CONFIG = {
    'score_reward_factor': 1.0/5.0,    # 得分奖励系数
    'death_penalty': -1.0,              # 死亡惩罚
    'kill_reward': 20.0,                # 击杀奖励
    'time_penalty': -0.05,              # 时间惩罚
    'level_complete_reward': 500.0,     # 通关奖励
    'game_over_penalty': -30.0,         # 游戏结束惩罚
    'action_reward': 0.1,               # 行动奖励
    'no_action_penalty': -0.3,          # 不动惩罚
    'fire_reward': 0.5,                 # 开火奖励
}
```

## 📊 训练结果

### 关卡 1 (DQN)

| 指标   | 数值        |
|------|-----------|
| 通关率  | **100%**  |
| 平均奖励 | **92.09** |
| 平均得分 | **233**   |

### 关卡 2 (PPO)

| 指标   | 数值        |
|------|-----------|
| 通关率  | **80%**   |
| 平均奖励 | **3452**  |
| 平均击杀 | **59**    |

### 训练曲线

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./logs
```

然后在浏览器打开 `http://localhost:6006`

## 🔧 高级用法

### 继续训练

```bash
# 自动从当前算法的已有模型继续训练
python train.py --algo ppo --level 2

# 指定训练步数
python train.py --algo ppo --level 2 --steps 10000000
```

### 课程学习

从简单关卡开始，迁移到更难关卡：

```bash
# 1. 先训练关卡 1
python train.py --algo ppo --level 1

# 2. 复制关卡 1 模型作为关卡 2 起点
cp models/level_1/best_model_ppo.zip models/level_2/best_model_ppo.zip

# 3. 继续训练关卡 2
python train.py --algo ppo --level 2
```

### 自定义奖励

修改 `config.py` 中的 `REWARD_CONFIG` 来调整奖励函数：

```python
# 更鼓励攻击性行为
REWARD_CONFIG['kill_reward'] = 25.0       # 增加击杀奖励
REWARD_CONFIG['fire_reward'] = 1.0        # 增加开火奖励
REWARD_CONFIG['death_penalty'] = -0.5     # 减少死亡惩罚
```

## 🎓 技术细节

### 环境设计

- 继承 `gymnasium.Env`，符合 Gym API 规范
- 支持无头模式（`SDL_VIDEODRIVER=dummy`）用于快速训练
- 每步对应约 20ms 游戏时间

### 网络结构

自定义 MlpPolicy：
- 2 层全连接网络
- 每层 **256** 个神经元（比默认更大）
- ReLU 激活函数

### 探索策略

DQN 使用 ε-greedy 策略：
- 初始 ε = 1.0，最终 ε = 0.05
- 探索比例占总训练步数的 20%

PPO 使用熵系数控制探索：
- ent_coef = 0.01

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
