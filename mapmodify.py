import numpy as np
from numba import jit
import random

def update_explored_map(explored_map, agent_position, obstacles, grid_config):
    x, y = agent_position
    obs_radius = grid_config.obs_radius
    # 遍历智能体的观察空间，并更新 explored_map
    for dx in range(-obs_radius, obs_radius + 1):
        for dy in range(-obs_radius, obs_radius + 1):
            nx, ny = x + dx, y + dy
            explored_map[nx, ny] = obstacles[nx, ny]
    
    return explored_map


def process_observations(grid_config, agents_positions, obstacles, explored_map):
    """
    处理智能体的观测数据，更新 explored_map 并打印结果
    """
    for idx, agents_position in enumerate(agents_positions):
        explored_map = update_explored_map(explored_map, agents_positions[idx], obstacles, grid_config)
        
    return explored_map


def detect_other_agents2(obs, grid_config):
    """
    检测当前观测空间中是否存在其他智能体（跳过中心点）
    
    参数:
    - obs: 当前的观测数据（包含智能体和其他信息）
    - grid_config: 包含观测空间的配置，例如 obs_radius
    
    返回值:
    - target_detected_list: 每个智能体的布尔值列表，True 表示检测到其他智能体，False 表示没有检测到
    """
    # 转换 obs[0] 为 NumPy 数组
    obs_array = np.array(obs)
    target_info_array = np.array([observation['agents'] for observation in obs])
    #print(target_info_array)
    # 初始化掩码，跳过中心点
    mask = np.ones((grid_config.obs_radius * 2 + 1, grid_config.obs_radius * 2 + 1), dtype=bool)
    mask[grid_config.obs_radius, grid_config.obs_radius] = False
    # 提取每个智能体的第 1 通道信息，并进行检测
    #print(obs[0])
    #target_info_array = obs_array[:, 1]
   
    # 使用向量化操作检测每个智能体是否检测到其他智能体（跳过中心点）
    target_detected_list = np.any(target_info_array[:, mask], axis=1)
    
    return target_detected_list


#代码存在问题
def detect_other_agents(obs, grid_config):
    """
    通过检测观测空间中智能体数量是否大于3来判断
    
    参数:
    - obs: 当前的观测数据（包含智能体和其他信息）
    - grid_config: 包含观测空间的配置，例如 obs_radius
    
    返回值:
    - target_detected_list: 每个智能体的布尔值列表，True 表示检测到其他智能体数量大于3，False 表示数量小于等于3
    """
    # 转换 obs[0] 为 NumPy 数组
    target_info_array = np.array([observation['agents'] for observation in obs])


    obs_info_array = np.array([observation['obstacles'] for observation in obs])
    
    
    # 初始化掩码，跳过中心点
    mask = np.ones((grid_config.obs_radius * 2 + 1, grid_config.obs_radius * 2 + 1), dtype=bool)
    mask[grid_config.obs_radius, grid_config.obs_radius] = False
    
    # 计算每个智能体观测空间内的智能体数量（排除中心点）
    agent_counts = np.sum(target_info_array[:, mask], axis=1)

    #obs_counts = np.sum(obs_info_array[:, mask], axis=1)
    #free_space = 48 - obs_counts
    # 判断智能体数量是否大于3
    #a = agent_counts/obs_counts
    #free = 1 - obs_counts
    target_detected_list = agent_counts > 3
    #print(target_detected_list)
    #print(agent_counts)
    return agent_counts
    #return target_detected_list

def get_action_from_displacement(dx, dy):
    if dx == 0 and dy == 0:
        #print("保持不动")
        return 0  # 保持不动
    elif dx == -1 and dy == 0:
        #print("向上移动")
        return 1  # 向上移动
    elif dx == 1 and dy == 0:
        #print("向下移动")
        return 2  # 向下移动
    elif dx == 0 and dy == -1:
        #print("向左移动")
        return 3  # 向左移动
    elif dx == 0 and dy == 1:
        #print("向右移动")
        return 4  # 向右移动
    else:
        raise ValueError("Invalid displacement: dx={}, dy={}".format(dx, dy))


import numpy as np
import numpy as np
from numba import jit


@jit(nopython=True)
def get_sensed_map(agents_positions, obstacles_map, terminated, radius=3):
    """
    获取每个智能体的独立 sensed_map，结合障碍物地图，将在观察范围半径内的其他智能体位置添加到各自的 sensed_map 中。
    
    :param agents_positions: 所有智能体的位置信息列表，列表中每个元素是 (x, y) 元组
    :param obstacles_map: 障碍物地图，二维 numpy 数组，0 表示空闲，1 表示障碍物
    :param radius: 智能体的观察范围半径，默认为 3
    :return: sensed_maps 列表，每个元素是一个智能体的 sensed_map
    """
    map_height, map_width = obstacles_map.shape
    sensed_maps = []

    # 将 agents_positions 转换为 NumPy 数组，便于向量化计算
    agents_positions = np.array(agents_positions)
    for i in range(len(agents_positions)):
        agent_x, agent_y = agents_positions[i]
        
        # 复制障碍物地图
        sensed_map = obstacles_map.copy()

        # 获取当前智能体的感知范围
        x_min, x_max = max(0, agent_x - radius), min(map_height, agent_x + radius + 1)
        y_min, y_max = max(0, agent_y - radius), min(map_width, agent_y + radius + 1)

        ## 筛选在该智能体感知范围内的其他智能体
        #for j in range(len(agents_positions)):
        #    if i != j or terminated[j]==True:  # 跳过自身
        #        other_x, other_y = agents_positions[j]
        #        if x_min <= other_x < x_max and y_min <= other_y < y_max:
        #            sensed_map[other_x, other_y] = 1
#
        sensed_maps.append(sensed_map)

    return sensed_maps

# 获取可移动位置的函数
def get_available_moves(explored_map, agent_position):
    x, y = agent_position
    available_moves = []
    # 定义四个可能的移动方向（上、下、左、右）
    moves = [
        (x - 1, y),  # 上
        (x + 1, y),  # 下
        (x, y - 1),  # 左
        (x, y + 1)   # 右
    ]
    action = 0
    # 遍历每个可能的移动方向
    for move in moves:
        nx, ny = move
        if explored_map[nx, ny] == 0:
           available_moves.append(move)
     # 随机选择一个可移动的位置
    if available_moves:
        target_position = random.choice(available_moves)
        dx = target_position[0] - agent_position[0]
        dy = target_position[1] - agent_position[1]
        action = get_action_from_displacement(dx, dy)
        return action
    else:
        target_position = random.choice(moves)
        dx = target_position[0] - agent_position[0]
        dy = target_position[1] - agent_position[1]
        action = get_action_from_displacement(dx, dy)
        return action
        # 如果没有可移动的位置，保持不动
    #    return get_action_from_displacement(0, 0)
def get_available_space(explored_map, agent_position):
    x, y = agent_position
    available_actions = []
    # 定义四个可能的移动方向（上、下、左、右）
    moves = [
        (x - 1, y),  # 上
        (x + 1, y),  # 下
        (x, y - 1),  # 左
        (x, y + 1)   # 右
    ]
    action = 0
    # 遍历每个可能的移动方向
    for move in moves:
        nx, ny = move
        if explored_map[nx, ny] == 0:
           dx = nx - agent_position[0]
           dy = ny - agent_position[1]
           action = get_action_from_displacement(dx, dy)
           available_actions.append(action)

    return available_actions
    

