import numpy as np
import time
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Manager, set_start_method
from pathlib import Path

from mapmodify import (
    process_observations,
    get_available_moves,
    get_available_space,
    get_sensed_map,
    get_action_from_displacement,
    detect_other_agents
)
from init_env import initialize_environment, reset_environment
from animation import AnimationMonitor

#from pymarl import model
from appo import model  # 如果确实使用了 appo 里的 EPOM，需要确认是否和 pymarl.model 冲突
from pydantic import BaseModel, Extra
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 3
    device: str = 'cuda'
# 初始化环境

class EpomConfig(AlgoBase, extra=Extra.forbid):
    path_to_weights: str = "weights/epom/checkpoint_p0"
    name: Literal['EPOM'] = 'EPOM'

algo = model.EPOM(EpomConfig(path_to_weights=str(Path('weights/epom'))))
# 环境初始化函数
def setup_environment(seed, set_density,  set_size, set_num_agents):
    env, env2, obs, explored_map, agents, grid_config, paths,explored_map_pathuse = initialize_environment(set_seed = seed, RL_Mode=True,  set_density =set_density,  set_size = set_size, set_num_agents = set_num_agents)
    obstacles = env.grid.get_obstacles()
    env = AnimationMonitor(env.env)
    env.reset_animation()
    # 状态跟踪初始化
    stationary_steps = {i: 0 for i in range(len(agents))}
    last_positions = [None] * len(agents)
    last2_positions = [None] * len(agents)
    terminated = [False] * len(agents)
    return env, obs, agents, grid_config, paths, last_positions, last2_positions, terminated, explored_map, obstacles,explored_map_pathuse

import random
def batch_agent_step(agents, current_positions, sensed_map, last_positions, last2_positions, paths, terminated, agent_counts, obs, explored_map, EPOMcount,DLitecount):
    actions = np.zeros(len(agents), dtype=int)
    actions3 = algo.act(obs, explored_map=explored_map, agents_positions=current_positions)
    for i, (agent, current_pos) in enumerate(zip(agents, current_positions)):
        if not terminated[i]:
            available_actions = get_available_space(explored_map, current_pos)
            if agent_counts[i]>4: 
               actions[i] = actions3[i] 
            else:
                paths[i] = None 
                agent.sensed_map = sensed_map[i].copy()
                actions[i], paths[i] = agent.global_planning(current_pos, paths[i])
                if paths[i] == None :
                   actions[i] = actions3[i] 
                else:
                    if  np.array_equal(current_pos, last2_positions[i]):
                        if agent_counts[i] !=0: 
                           actions[i] = actions3[i]   
                        else: 
                           actions[i] = random.choice(available_actions)
 
           #一个方式是共享地图，共享地图的策略也会导致智能体尝试寻找新的路。
           
           #增加一个机制，检测智能体在动作选择前避免冲突，然后其他baseline都是会地图感知的包括PICO SCHRIMP、DCC、DHC、


           #需要测试的地方， sensed map 是不是需要修改，还是不要了

           #暴力机制，就是检测所有智能体如果导致无法移动就选择另外的。
           #以及机制，当检测智能体需要交换位置或者
           #一个机制可以设置成智能体只能走两次
           #还有个问题就是其他算法貌似是
            if actions[i] not in available_actions and actions[i]!=0:
               actions[i] = random.choice(available_actions)
            # 如果 actions[i] 是一个 NumPy 数组，需要先转换为元组进行比较

            last2_positions[i] = last_positions[i]
            last_positions[i] = current_pos

    return actions, paths
    
import multiprocessing
import pandas as pd
import time
import numpy as np


def run_experiment(seed, results_list, set_density, set_size,set_num_agents):
    env, obs, agents, grid_config, paths, last_positions, last2_positions, terminated, explored_map, obstacles, explored_map_pathuse = setup_environment(seed,  set_density,  set_size, set_num_agents)
    start_time = time.time()
    actions = np.zeros(len(agents), dtype=int)
    actions3 = np.zeros(len(agents), dtype=int)
    step = 0
    EPOMcount = 0
    DLitecount = 0
    while True:
        step += 1
        target_detected_list = detect_other_agents(obs, grid_config)
        sensed_maps = get_sensed_map(env.grid.get_agents_xy(), explored_map, terminated)
        agents_positions = env.get_agents_xy()
        actions, paths = batch_agent_step(agents, agents_positions, sensed_maps, last_positions, last2_positions, paths, terminated, target_detected_list,obs, obstacles, EPOMcount,DLitecount)
        obs, rewards, terminated, infos = env.step(actions)
        if all(terminated):
            elapsed_time = time.time() - start_time
            metrics = infos[0].get('metrics')
            results_list.append({
                "seed" : seed,
                "step": metrics['step'],
                "ISR": metrics['ISR'],
                "CSR": metrics['CSR'],
            })
            
            break
        
        agents_positions = env.get_agents_xy()
        explored_map = process_observations(grid_config, agents_positions, obstacles, explored_map)
        explored_map_pathuse  = process_observations(grid_config, agents_positions, obstacles, explored_map_pathuse)
if __name__ == "__main__":
    start_time_all = time.time()
    seeds = list(range(1,2))  # 1到50的实验
    multiprocessing.set_start_method('spawn')

    num_agents_list = [8]  # 现在表示智能体数量
    grid_size = 15 # 现在表示固定的网格大小

    densities = [0.15]

    summary_data = []  # 用于存储所有实验的平均 step 和 CSR

    with multiprocessing.Manager() as manager:
        #results_list = manager.list()  # 共享的结果列表

        # 遍历不同智能体数量和密度
        for density in densities:
            for num_agents in num_agents_list:
                results_list = manager.list() 
                print(f"Running experiments for density={density}, num_agents={num_agents}, grid_size={grid_size}x{grid_size}")

                # 使用多进程池
                with multiprocessing.Pool(processes=1) as pool:
                    pool.starmap(run_experiment, [(seed, results_list, density, grid_size, num_agents) for seed in seeds])

                # 将结果转换为 DataFrame
                results_df = pd.DataFrame(list(results_list))

                # 计算平均 step 和 CSR
                avg_step = results_df['step'].mean()
                avg_csr = results_df['CSR'].mean()

                # 在 DataFrame 中添加 avg_step 和 avg_csr
                results_df['avg_step'] = avg_step
                results_df['avg_csr'] = avg_csr

                # 保存详细实验结果到 CSV
                filename = f"experiment_results_{density}_grid{grid_size}x{grid_size}_agents{num_agents}.csv"
                results_df.to_csv(filename, index=False)

                # 记录到 summary_data
                summary_data.append({
                    'Density': density,
                    'Grid Size': f"{grid_size}x{grid_size}",
                    'Num Agents': num_agents,
                    'Average Step': avg_step,
                    'Average CSR': avg_csr
                })

                # 打印结果
                print(f"Density: {density}, Grid Size: {grid_size}x{grid_size}, Num Agents: {num_agents}")
                print("平均 step:", avg_step)
                print("平均 CSR:", avg_csr)

    # 保存 summary_data 到总表格
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("summary_results1.csv", index=False)

    end_time_all = time.time()
    print("所有实验完成，总耗时:", end_time_all - start_time_all, "秒")
    print("所有实验的平均 step 和 CSR 结果已保存到 'summary_results.csv'.")

