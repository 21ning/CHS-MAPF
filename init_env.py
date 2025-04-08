from pogema.integrations.make_pogema import pogema_v0
from pogema.grid_config import GridConfig
from mapmodify import  process_observations, detect_other_agents
import numpy as np
from dstar_lite import DLitePlanner  # 只导入 DLitePlanner
import random



def initialize_environment(set_seed = None, RL_Mode = False, set_density = 0.15, set_size = 128, set_num_agents=32):
    max_episode_steps2 = 128
    # 初始化 Pogema 环境
    if set_size == 128:
       max_episode_steps2 = 512
    elif set_size == 80:
       max_episode_steps2 = 480
    elif set_size == 40:
       max_episode_steps2 = 320
    elif set_size == 20:
       max_episode_steps2 = 256
    #if 
    grid_config = GridConfig(num_agents = set_num_agents, size= set_size, density= set_density, seed = set_seed , max_episode_steps = max_episode_steps2, obs_radius = 3)
    env = pogema_v0(grid_config=grid_config)
    env2 = None
    if RL_Mode == True:
        grid_config.integration="SampleFactory"
        grid_config.observation_type = "POMAPF"
        env2 = pogema_v0(grid_config=grid_config)
    #是否开启动画录制
    #env = AnimationMonitor(env, AnimationConfig())
    # obs[0][agentidx][aisle]
    # 获取地图信息，智能体观测数据的第0通道（障碍物信息），智能体观测数据的第1通道（智能体信息），智能体观测数据的第2通道（目标点）
    obs = env.reset()
    obstacles = env.grid.get_obstacles(ignore_borders=False)
    agents_positions = env.get_agents_xy()  # 获取智能体的位置
    explored_map = process_observations(grid_config, agents_positions, obstacles, np.zeros_like(obstacles))
    
    explored_map_pathuse = process_observations(grid_config, agents_positions, obstacles, np.ones_like(obstacles) )
    #print(env.grid.get_obstacles(ignore_borders=True))
    goal_reached = 0
    count = 0
    # 初始化 D* Lite 路径规划器
    agents = []
    for i in range(grid_config.num_agents):
        start_pos = tuple(env.grid.get_agents_xy()[i])
        target_pos = tuple(env.grid.get_targets_xy()[i])
        planner = DLitePlanner(sensed_map=explored_map, start=start_pos, goal=target_pos)
        #planner = dlite_planner.DLitePlanner(explored_map.astype(int).tolist(), start_pos, target_pos)

        agents.append(planner)
    paths = [[] for _ in range(grid_config.num_agents)]
    for i, agent in enumerate(agents):
        current_pos = tuple(env.grid.get_agents_xy()[i])
        paths[i] = agent.plan(current_pos)
    return env, env2, obs, explored_map, agents, grid_config, paths,explored_map_pathuse



def reset_environment(env, grid_config):
    # 初始化 Pogema 环境
    #obs = env.reset(seed=random.randint(1, 10000))[0]
    obs = env.reset()[0]
    obstacles = env.grid.get_obstacles()
    agents_positions = env.get_agents_xy()  # 获取智能体的位置
    explored_map = process_observations(grid_config, agents_positions, obs, np.zeros_like(obstacles))
    agents = []
    for i in range(grid_config.num_agents):
        start_pos = tuple(env.grid.get_agents_xy()[i])
        target_pos = tuple(env.grid.get_targets_xy()[i])
        planner = DLitePlanner(sensed_map=explored_map, start=start_pos, goal=target_pos)
        agents.append(planner)
    paths = [[] for _ in range(grid_config.num_agents)]
    for i, agent in enumerate(agents):
        current_pos = tuple(env.grid.get_agents_xy()[i])
        paths[i] = agent.plan(current_pos)
    return env, obs, explored_map, agents, grid_config, paths




def reset_environment(env, grid_config):
    # 初始化 Pogema 环境
    #obs = env.reset(seed=random.randint(1, 10000))[0]
    obs = env.reset()[0]
    obstacles = env.grid.get_obstacles()
    agents_positions = env.get_agents_xy()  # 获取智能体的位置
    explored_map = process_observations(grid_config, agents_positions, obs, np.zeros_like(obstacles))
    agents = []
    for i in range(grid_config.num_agents):
        start_pos = tuple(env.grid.get_agents_xy()[i])
        target_pos = tuple(env.grid.get_targets_xy()[i])
        planner = DLitePlanner(sensed_map=explored_map, start=start_pos, goal=target_pos)
        agents.append(planner)
    paths = [[] for _ in range(grid_config.num_agents)]
    for i, agent in enumerate(agents):
        current_pos = tuple(env.grid.get_agents_xy()[i])
        paths[i] = agent.plan(current_pos)
    return env, obs, explored_map, agents, grid_config, paths


def initialize_environment2222():
    # 初始化 Pogema 环境
    grid_config = GridConfig(num_agents=128, size=34, density=0.3, obs_radius=3,integration="PyMARL")
    env = pogema_v0(grid_config=grid_config)
    #是否开启动画录制
    #env = AnimationMonitor(env, AnimationConfig())
    # obs[0][agentidx][aisle]
    # 获取地图信息，智能体观测数据的第0通道（障碍物信息），智能体观测数据的第1通道（智能体信息），智能体观测数据的第2通道（目标点）
    obs = env.get_obs()  # 检查观测值的形状
    # 计数智能体的观测数量
    #num_agents = len(obs[0])
    print(obs)
    obs2 = env.get_state()
    #print(obs2)

        # Assuming `env` is an instance of your environment class
    env_info = env.get_env_info()
    
    # Accessing the individual elements from env_info
    state_shape = env_info["state_shape"]
    print("__________")
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_limit = env_info["episode_limit"]

    # Printing or using these variables
    print(f"State Shape: {state_shape}")
    print(f"Observation Shape: {obs_shape}")
    print(f"Number of Actions: {n_actions}")
    print(f"Number of Agents: {n_agents}")
    print(f"Episode Limit: {episode_limit}")


    grid = env.env.grid
    obstacles = grid.get_obstacles()
    agents_positions = grid.get_agents_xy()  # 获取智能体的位置
    #explored_map = process_observations(grid_config, agents_positions, obs, np.zeros_like(obstacles))
    goal_reached = 0
    count=0
    #env.render() 
    # 初始化 D* Lite 路径规划器
    agents = []
    explored_map = []
    paths = [[] for _ in range(grid_config.num_agents)]
    for i, agent in enumerate(agents):
        current_pos = tuple(grid.get_agents_xy()[i])
        paths[i] = agent.plan(current_pos)
    return env, env2, obs, explored_map, agents, grid_config, paths
