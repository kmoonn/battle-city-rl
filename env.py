#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坦克大战 Gym 环境包装器 - 简化版
用于 Stable-Baselines3 强化学习训练
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BattleCityEnv(gym.Env):
    """坦克大战强化学习环境"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # 瓦片类型
    TILE_EMPTY = 0
    TILE_BRICK = 1
    TILE_STEEL = 2
    TILE_WATER = 3
    TILE_GRASS = 4
    TILE_FROZE = 5

    # 方向
    DIR_UP = 0
    DIR_RIGHT = 1
    DIR_DOWN = 2
    DIR_LEFT = 3

    def __init__(self, render_mode=None, level=1):
        super().__init__()

        self.render_mode = render_mode
        self.level = level

        # 如果不是渲染模式，设置无头模式
        if render_mode is None:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # 动作空间: 0-无操作, 1-上, 2-右, 3-下, 4-左, 5-开火
        self.action_space = spaces.Discrete(6)

        # 扁平观察空间 (727维)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(727,), dtype=np.float32
        )

        # 游戏状态
        self.game_state = None
        self.game_over = False
        self.level_complete = False
        self.step_count = 0
        self.max_steps = 10000

        # 奖励跟踪
        self.last_score = 0
        self.last_lives = 3
        self.last_enemies_count = 0
        self.total_kills = 0  # 跟踪总击杀数

        # 关卡敌人配置
        self.levels_enemies = (
            (18,2,0,0), (14,4,0,2), (14,4,0,2), (2,5,10,3), (8,5,5,2),
            (9,2,7,2), (7,4,6,3), (7,4,7,2), (6,4,7,3), (12,2,4,2),
            (5,5,4,6), (0,6,8,6), (0,8,8,4), (0,4,10,6), (0,2,10,8),
            (16,2,0,2), (8,2,8,2), (2,8,6,4), (4,4,4,8), (2,8,2,8),
            (6,2,8,4), (6,8,2,4), (0,10,4,6), (10,4,4,2), (0,8,2,10),
            (4,6,4,6), (2,8,2,8), (15,2,2,1), (0,4,10,6), (4,8,4,4),
            (3,8,3,6), (6,4,2,8), (4,4,4,8), (0,10,4,6), (0,6,4,10)
        )

    def _create_game_state(self):
        """创建游戏状态对象"""
        import tanks

        # 清理全局状态
        del tanks.players[:]
        del tanks.enemies[:]
        del tanks.bullets[:]
        del tanks.bonuses[:]
        del tanks.labels[:]
        del tanks.gtimer.timers[:]

        # 创建游戏实例
        game = tanks.Game()
        game.game_over = False
        game.running = True
        game.active = True

        # 创建关卡
        level_nr = self.level
        game.level = tanks.Level(level_nr)
        game.stage = level_nr

        # 设置敌人配置
        if level_nr <= 35:
            enemies_l = self.levels_enemies[level_nr - 1]
        else:
            enemies_l = self.levels_enemies[34]

        game.level.enemies_left = [0]*enemies_l[0] + [1]*enemies_l[1] + [2]*enemies_l[2] + [3]*enemies_l[3]
        random.shuffle(game.level.enemies_left)

        # 重置基地
        tanks.castle.rebuild()

        # 创建玩家
        x = 8 * 16 + (16 * 2 - 26) / 2
        y = 24 * 16 + (16 * 2 - 26) / 2
        player = tanks.Player(game.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2))
        player.lives = 3
        player.score = 0
        player.trophies = {"bonus": 0, "enemy0": 0, "enemy1": 0, "enemy2": 0, "enemy3": 0}
        player.state = tanks.Player.STATE_ALIVE
        tanks.players.append(player)

        return game

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 创建游戏状态
        self.game_state = self._create_game_state()
        self.game_over = False
        self.level_complete = False
        self.step_count = 0

        # 奖励跟踪
        self.last_score = 0
        self.last_lives = 3
        self.last_enemies_count = len(self.game_state.level.enemies_left)

        # 初始步进（跳过生成动画）
        for _ in range(30):
            self._step_game(None)

        return self._get_observation(), {'level': self.level}

    def step(self, action):
        self._step_game(action)
        self.step_count += 1

        # 计算奖励
        reward = self._calculate_reward()

        # 检查是否结束
        done = self._is_done()

        info = {
            'score': self._get_player_score(),
            'lives': self._get_player_lives(),
            'enemies_remaining': self._get_enemies_remaining(),
            'level_complete': self.level_complete,
            'step': self.step_count,
            'total_kills': self.total_kills
        }

        return self._get_observation(), reward, done, False, info

    def _step_game(self, action):
        """执行游戏步进"""
        import tanks

        game = self.game_state
        time_passed = 20

        # 处理玩家动作
        if tanks.players and tanks.players[0].state == tanks.Player.STATE_ALIVE:
            player = tanks.players[0]
            if action is not None:
                if action == 1:
                    player.move(self.DIR_UP)
                elif action == 2:
                    player.move(self.DIR_RIGHT)
                elif action == 3:
                    player.move(self.DIR_DOWN)
                elif action == 4:
                    player.move(self.DIR_LEFT)
                elif action == 5:
                    player.fire()

        # 更新玩家
        for player in tanks.players:
            player.update(time_passed)

        # 更新敌人
        for enemy in tanks.enemies[:]:
            if enemy.state == tanks.Enemy.STATE_DEAD:
                tanks.enemies.remove(enemy)
            else:
                enemy.update(time_passed)

        # 检查玩家状态
        for player in tanks.players:
            if player.state == tanks.Player.STATE_DEAD:
                player.lives -= 1
                if player.lives > 0:
                    game.respawnPlayer(player)
                else:
                    self.game_over = True

        # 更新子弹
        for bullet in tanks.bullets[:]:
            if bullet.state == tanks.Bullet.STATE_REMOVED:
                tanks.bullets.remove(bullet)
            else:
                bullet.update()

        # 更新道具
        for bonus in tanks.bonuses[:]:
            if not bonus.active:
                tanks.bonuses.remove(bonus)

        # 检查基地
        if not tanks.castle.active:
            self.game_over = True

        # 更新定时器
        tanks.gtimer.update(time_passed)

        # 生成敌人
        game.spawnEnemy()

        # 检查通关
        if len(game.level.enemies_left) == 0 and len(tanks.enemies) == 0:
            self.level_complete = True

    def _get_observation(self):
        import tanks

        obs = np.zeros(727, dtype=np.float32)
        offset = 0

        # 玩家状态 (4)
        if tanks.players and tanks.players[0].state == tanks.Player.STATE_ALIVE:
            player = tanks.players[0]
            obs[0:4] = [
                player.rect.x / 416.0,
                player.rect.y / 416.0,
                player.direction / 4.0,
                player.health / 100.0
            ]
        offset += 4

        # 敌人状态 (16 = 4 * 4)
        for i, enemy in enumerate(tanks.enemies[:4]):
            obs[offset + i*4 : offset + i*4 + 4] = [
                enemy.rect.x / 416.0,
                enemy.rect.y / 416.0,
                enemy.direction / 4.0,
                (enemy.type + 1) / 4.0
            ]
        offset += 16

        # 子弹状态 (30 = 10 * 3)
        for i, bullet in enumerate(tanks.bullets[:10]):
            obs[offset + i*3 : offset + i*3 + 3] = [
                bullet.rect.x / 416.0,
                bullet.rect.y / 416.0,
                bullet.direction / 4.0
            ]
        offset += 30

        # 地图状态 (676 = 26 * 26)
        if self.game_state and self.game_state.level:
            for tile in self.game_state.level.mapr:
                grid_x = int(tile.x // 16)
                grid_y = int(tile.y // 16)
                if 0 <= grid_x < 26 and 0 <= grid_y < 26:
                    obs[offset + grid_y * 26 + grid_x] = tile.type / 6.0
        offset += 676

        # 剩余敌人 (1)
        obs[offset] = self._get_enemies_remaining() / 20.0

        return obs

    def _get_player_score(self):
        import tanks
        if tanks.players:
            return tanks.players[0].score
        return 0

    def _get_player_lives(self):
        import tanks
        if tanks.players:
            return tanks.players[0].lives
        return 0

    def _get_enemies_remaining(self):
        import tanks
        if self.game_state and self.game_state.level:
            return len(self.game_state.level.enemies_left) + len(tanks.enemies)
        return 0

    def _calculate_reward(self):
        """计算奖励值 - 更鼓励攻击性行为"""
        reward = 0.0

        # 得分奖励 (击杀敌人、收集道具) - 更大的奖励
        current_score = self._get_player_score()
        score_diff = current_score - self.last_score
        if score_diff > 0:
            reward += score_diff / 30.0  # 进一步增加奖励
        self.last_score = current_score

        # 生命惩罚 - 中等惩罚
        current_lives = self._get_player_lives()
        if current_lives < self.last_lives:
            reward -= 1.5
        self.last_lives = current_lives

        # 击杀敌人奖励 - 更大的奖励
        current_enemies = self._get_enemies_remaining()
        if current_enemies < self.last_enemies_count:
            kills = self.last_enemies_count - current_enemies
            reward += 2.0 * kills  # 增加击杀奖励
            self.total_kills += kills  # 跟踪总击杀数
        self.last_enemies_count = current_enemies

        # 时间惩罚 - 鼓励快速行动
        reward -= 0.01

        # 移除生存奖励 - 不要鼓励"苟活"

        # 通关奖励 - 非常大的奖励
        if self.level_complete:
            reward += 50.0

        # 游戏结束惩罚
        if self.game_over and not self.level_complete:
            reward -= 5.0

        return reward

    def _is_done(self):
        return self.game_over or self.level_complete or self.step_count >= self.max_steps

    def render(self):
        """渲染游戏画面"""
        import pygame

        if self.render_mode == "human" and self.game_state:
            # 调用游戏的 draw 方法
            self.game_state.draw()
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array" and self.game_state:
            # 返回 RGB 数组
            import tanks
            if hasattr(self.game_state, 'screen') and self.game_state.screen:
                return pygame.surfarray.array3d(self.game_state.screen).transpose(1, 0, 2)
        return None

    def close(self):
        import pygame
        pygame.quit()

    def get_action_meanings(self):
        return ['NOOP', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'FIRE']


# 测试代码
if __name__ == "__main__":
    print("测试 BattleCityEnv...")

    env = BattleCityEnv(render_mode=None, level=1)

    obs, info = env.reset()
    print(f"观察空间形状: {obs.shape}")
    print(f"初始信息: {info}")

    print("\n测试动作...")
    total_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"步 {i+1}: 动作={action}, 奖励={reward:.3f}, 累计奖励={total_reward:.3f}")

        if done:
            print(f"回合结束! 原因: {'通关' if info['level_complete'] else '游戏结束'}")
            break

    env.close()
    print("\n环境测试完成!")
