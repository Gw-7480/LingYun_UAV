import gym
from gym import spaces
import numpy as np


class DroneAvoidanceEnv(gym.Env):
    def __init__(self, grid_size=10, num_obstacles=15):
        super(DroneAvoidanceEnv, self).__init__()

        self.grid_size = grid_size
        self.num_obstacles = num_obstacles

        # 定义动作空间：上下左右和对角线移动
        self.action_space = spaces.Discrete(8)  # [0-3]为上下左右, [4-7]为对角线方向

        # 定义状态空间：无人机位置在网格中，展平为一维
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size * grid_size + 2,), dtype=np.int32)

        self.start_position = np.array([0, 0], dtype=np.int32)  # 左上角
        self.target_position = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)  # 右下角
        self.drone_position = self.start_position.copy()
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        # 在网格中随机选择障碍物位置
        obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        indices = np.random.choice(self.grid_size * self.grid_size, self.num_obstacles, replace=False)
        np.put(obstacles, indices, 1)  # 设置障碍物为1
        obstacles[tuple(self.start_position)] = 0  # 确保起始位置没有障碍物
        obstacles[tuple(self.target_position)] = 0  # 确保目标位置没有障碍物
        return obstacles

    def reset(self):
        self.drone_position = self.start_position.copy()
        self.obstacles = self._generate_obstacles()
        return self._get_observation()

    def _get_observation(self):
        # 将无人机位置和障碍物展平为一个一维数组
        obs = np.zeros(self.grid_size * self.grid_size, dtype=np.int32)
        obs[self.grid_size * self.drone_position[0] + self.drone_position[1]] = 1  # 标记无人机位置
        obs = np.append(obs, [self.drone_position[0], self.drone_position[1]])  # 添加无人机坐标
        obs[-self.grid_size * self.grid_size:] = self.obstacles.flatten()  # 添加障碍物
        return obs

    def step(self, action):
        direction_vectors = [
            np.array([0, 1]),  # 下
            np.array([0, -1]),  # 上
            np.array([1, 0]),  # 右
            np.array([-1, 0]),  # 左
            np.array([1, 1]),  # 右下
            np.array([-1, -1]),  # 左上
            np.array([1, -1]),  # 右上
            np.array([-1, 1])  # 左下
        ]

        new_position = self.drone_position + direction_vectors[action]
        new_position = np.clip(new_position, [0, 0], [self.grid_size - 1, self.grid_size - 1])

        reward = 0
        done = False

        # 检查是否撞到障碍物
        if self.obstacles[tuple(new_position)] == 1:
            reward -= 100
            self.drone_position = self.start_position.copy()  # 重置到起始位置
            done = False
        else:
            self.drone_position = new_position

        # 检查是否到达目标位置
        if np.array_equal(self.drone_position, self.target_position):
            reward += 10
            done = True

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = '.'
        grid[self.obstacles == 1] = 'X'  # 标记障碍物
        grid[tuple(self.target_position)] = 'T'  # 标记目标位置
        grid[tuple(self.drone_position)] = 'D'  # 标记无人机位置
        print("\n".join(["".join(row) for row in grid]))

    def close(self):
        pass


if __name__ == '__main__':
    # 使用环境
    env = DroneAvoidanceEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # 随机选择一个动作
        obs, reward, done, _ = env.step(action)
        env.render()

    env.close()
