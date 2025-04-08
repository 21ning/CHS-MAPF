import gym
import numpy as np
from gym.spaces import Box
from pogema import GridConfig
from pogema.grid import Grid


class GridMemory:
    #修改为地图所需要的尺寸
    def __init__(self, start_r=32):
        self.memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1))

    @staticmethod
    def try_to_insert(x, y, source, target):
        r = source.shape[0] // 2
        try:
            target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except ValueError:
            return False

    def increase_memory(self):
        m = self.memory
        r = self.memory.shape[0]
        self.memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        assert self.try_to_insert(r, r, m, self.memory)

    #根据xy移动的坐标以及障碍物将障碍物插入到动态网格里
    def update(self, x, y, obstacles):
        while True:
            r = self.memory.shape[0] // 2
            if self.try_to_insert(r + x, r + y, obstacles, self.memory):
                break
            self.increase_memory()

    #获取指定位置的记忆网格
    def get_observation(self, x, y, obs_radius):
        while True:
            r = self.memory.shape[0] // 2
            tx, ty = x + r, y + r
            size = self.memory.shape[0]
            if 0 <= tx - obs_radius and tx + obs_radius + 1 <= size:
                if 0 <= ty - obs_radius and ty + obs_radius + 1 <= size:
                    return self.memory[tx - obs_radius:tx + obs_radius + 1, ty - obs_radius:ty + obs_radius + 1]

            self.increase_memory()

    def render(self):
        m = self.memory.astype(int).tolist()
        gc = GridConfig(map=m)
        g = Grid(add_artificial_border=False, grid_config=gc)
        r = self.memory.shape[0] // 2
        g.positions_xy = [[r, r]]
        g.finishes_xy = []
        g.render()

#修改代码，智能体初始位置是在当前地方。。。

#现在存在的问题是智能体的神经网络好像没见过这种类型的，可以分享
class MultipleGridMemory:
    def __init__(self):
        self.memories = None
        self.initpositions = None
    def update(self, observations, agents_positions, explored_map):
        # 初始化 memories 和初始位置
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [GridMemory() for _ in range(len(observations))]
            self.initpositions = agents_positions

        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update(*obs['xy'], obs['obstacles'])
            #直接相当于自己移动到了别人的位置不久可以了
            # 获取当前智能体的位置
            
            #现在该机器人人的位置 - 机器人初始位置
            
            #print(obs['xy'])
            
            # 遍历所有 memories
            #for memory_idx, memory in enumerate(self.memories):
#
            #    #当前遍历其他的机器人位置
            #    current_x, current_y = agents_positions[memory_idx]
#
            #    #原来机器人初始位置
            #    initial_x, initial_y = self.initpositions[agent_idx]
            #    # 计算相对坐标
            #    relative_x = current_x - initial_x
            #    relative_y = current_y - initial_y
            #    # print(relative_x,relative_y)
            #    # 更新当前 memory 网格
            #    #self.memories[agent_idx].update(relative_x, relative_y, obs['obstacles'])
    #获取所有的网格
    def get_observations(self, xy_list, obs_radius):
        return [self.memories[idx].get_observation(x, y, obs_radius) for idx, (x, y) in enumerate(xy_list)]

    #思路就是所有智能体共享一个地图
    def modify_observation(self, observations, obs_radius, explored_map):
        all_xy = [observations[idx]['xy'] for idx in range(len(observations))]
        #观察半径
        r = obs_radius
        #网格大小半径
        rr = observations[0]['agents'].shape[0] // 2
        for obs, gm_obs in zip(observations, self.get_observations(all_xy, obs_radius)):
            #print(all_xy)
            obs['obstacles'] = gm_obs
            #obs['obstacles'] = explored_map

        #给其他两个观测值拼接成一个相同尺寸的观测值
        for agent_idx, obs in enumerate(observations):
            if rr <= r:
                agents = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
                agents[r - rr:r + rr + 1, r - rr: r + rr + 1] = obs['agents']
                obs['agents'] = agents
            else:
                obs['agents'] = obs['agents'][rr - r:rr + r + 1, rr - r: rr + r + 1]
    def clear(self):
        self.memories = None


class GridMemoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_radius):
        super().__init__(env)
        self.obs_radius = obs_radius

        size = self.obs_radius * 2 + 1
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(size, size)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(size, size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

        self.mgm = MultipleGridMemory()

    def observation(self, observations):
        self.mgm.update(observations)
        self.mgm.modify_observation(observations, self.obs_radius)
        
        return observations

    def reset(self):
        self.mgm.clear()
        return self.observation(self.env.reset())
