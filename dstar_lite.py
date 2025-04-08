import numpy as np
import heapq
import random
from mapmodify import   get_action_from_displacement
INF = 1e7

class Element:
    def __init__(self, key, value1, value2):
        self.key = key
        self.value1 = value1
        self.value2 = value2

    def __lt__(self, other):
        return (self.value1, self.value2) < (other.value1, other.value2)

class DLitePlanner:
    def __init__(self, sensed_map, start, goal):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.k_m = 0
        self.rhs = np.ones_like(sensed_map) * INF
        self.g = np.copy(self.rhs)
        self.sensed_map = np.copy(sensed_map)
        self.rhs[tuple(self.goal)] = 0
        self.queue = []
        heapq.heappush(self.queue, Element(self.goal, *self.calculate_key(self.goal)))
        self.loop_detection = set()

    def calculate_key(self, node):
        g_rhs_min = min(self.g[tuple(node)], self.rhs[tuple(node)])
        return [g_rhs_min + self.h_estimate(self.start, node) + self.k_m, g_rhs_min]

    def update_vertex(self, u):
        if not np.array_equal(u, self.goal):
            min_rhs = INF
            for s in self.succ(u):
                if self.sensed_map[tuple(s)] != 1:
                    min_rhs = min(min_rhs, self.cost(u, s) + self.g[tuple(s)])
            self.rhs[tuple(u)] = min_rhs

        self.queue = [el for el in self.queue if not np.array_equal(el.key, u)]
        heapq.heapify(self.queue)

        if self.g[tuple(u)] != self.rhs[tuple(u)]:
            heapq.heappush(self.queue, Element(u, *self.calculate_key(u)))

    def compute_shortest_path(self):
        while self.queue and (heapq.nsmallest(1, self.queue)[0] < Element(self.start, *self.calculate_key(self.start)) or self.rhs[tuple(self.start)] != self.g[tuple(self.start)]):
            u = heapq.heappop(self.queue).key
            if self.g[tuple(u)] > self.rhs[tuple(u)]:
                self.g[tuple(u)] = self.rhs[tuple(u)]
                for s in self.succ(u):
                    self.update_vertex(s)
            else:
                self.g[tuple(u)] = INF
                for s in self.succ(u) + [u]:
                    self.update_vertex(s)

    def succ(self, u):
        directions = [np.array([i, j]) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i == 0) != (j == 0)]
        return [u + d for d in directions if 0 <= u[0] + d[0] < self.sensed_map.shape[0] and 0 <= u[1] + d[1] < self.sensed_map.shape[1]]

    def h_estimate(self, s1, s2):
        return np.sum(np.abs(np.array(s1) - np.array(s2)))

    def cost(self, u1, u2):
        return self.h_estimate(u1, u2) 

    def reset_partial(self):
        self.k_m = 0
        self.g = np.ones_like(self.sensed_map) * INF
        self.rhs = np.copy(self.g)
        self.rhs[tuple(self.goal)] = 0
        self.queue = []
        heapq.heappush(self.queue, Element(self.goal, *self.calculate_key(self.goal)))
        self.loop_detection = set()


    def plan(self, current_pos):
        self.start = np.array(current_pos)
        self.compute_shortest_path()
        path = []
        next_node = self.start
        visited = set()
        iteration = 0
        while not np.array_equal(next_node, self.goal): 
            iteration += 1
            path.append(tuple(next_node))
            visited.add(tuple(next_node))

            next_options = [tuple(n) for n in self.succ(next_node) if self.sensed_map[tuple(n)] != 1 and self.g[tuple(n)] < INF and tuple(n) not in visited]

            if not next_options:
                next_options2 = [tuple(n) for n in self.succ(next_node) if self.sensed_map[tuple(n)] != 1 and tuple(n) not in visited]
                if next_options2:
                    next_node = random.choice(next_options2)
                    continue

                break
            next_node = min(next_options, key=lambda x: self.g[x] + self.h_estimate(np.array(x), self.goal))
        if np.array_equal(next_node, self.goal):
            path.append(tuple(self.goal))
            #path = None
        #print(path)
        return path
        
    def is_path_valid(self, path):
        return all(self.sensed_map[node] != 1 for node in path)



    def global_planning(self, current_pos, paths):
        if paths is None:
            self.reset_partial()
            paths = self.plan(current_pos)
            #print(paths)
            if current_pos == tuple(self.goal):
                action = 0  # 目标已到达，无需进一步行动
            else:
                if paths == None or(len(paths) <= 1) :
                    action = 0
                    paths = None
                else:
                    next_step = paths[1]
                    dx, dy = next_step[0] - current_pos[0], next_step[1] - current_pos[1]
                    action = get_action_from_displacement(dx, dy)
                    paths.pop(0)

        elif len(paths) > 1:
        #if paths is not None :
            #障碍物调整问题
            for idx, next_path in enumerate(paths):
                # 如果发现障碍物，重新规划路径, 跳出循环重新处理路径
                if self.sensed_map[next_path[0], next_path[1]] == 1:
                    self.reset_partial()
                    paths = self.plan(current_pos)
                    break
            #找不到路径
            if (len(paths) <= 1):
                action = 0
                paths = None
            else:
                next_step = paths[1]
                dx, dy = next_step[0] - current_pos[0], next_step[1] - current_pos[1]
                action = get_action_from_displacement(dx, dy)
                paths.pop(0)
        else:
            # 当路径只剩下一个节点时，判断是否到达目标点
            if current_pos == tuple(self.goal):
                action = 0  # 目标已到达，无需进一步行动
            else:
                self.reset_partial()
                paths = self.plan(current_pos)
                if (len(paths) <= 1):
                    action = 0
                    paths = None
                else:
                    next_step = paths[1]
                    dx, dy = next_step[0] - current_pos[0], next_step[1] - current_pos[1]
                    action = get_action_from_displacement(dx, dy)
                    paths.pop(0) 
        return action, paths