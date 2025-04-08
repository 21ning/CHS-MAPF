#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>

namespace py = pybind11;

// 先放入C++版本的 DLitePlanner 和 global_planning 相关代码
// ---以下为简化的示例---

// 和之前相同的 get_action_from_displacement
int get_action_from_displacement(int dx, int dy) {
    if(dx == 0 && dy == 0) {
        return 0;
    } else if(dx == -1 && dy == 0) {
        return 1;
    } else if(dx == 0 && dy == 1) {
        return 2;
    } else if(dx == 1 && dy == 0) {
        return 3;
    } else if(dx == 0 && dy == -1) {
        return 4;
    }
    return 0;
}

static const double INF = 1e7;

struct Element {
    std::pair<int,int> key;
    double value1;
    double value2;
    Element(const std::pair<int,int>& k, double v1, double v2)
        : key(k), value1(v1), value2(v2) {}

    bool operator<(const Element& other) const {
        // 小根堆
        if(value1 == other.value1) {
            return value2 > other.value2;
        }
        return value1 > other.value1;
    }
};

class DLitePlanner {
public:
    // 构造函数
    DLitePlanner(const std::vector<std::vector<int>>& sensed_map,
                 std::pair<int,int> start,
                 std::pair<int,int> goal)
    {
        rows_ = (int)sensed_map.size();
        cols_ = (int)sensed_map[0].size();
        start_ = start;
        goal_ = goal;
        k_m_ = 0.0;

        g_.resize(rows_, std::vector<double>(cols_, INF));
        rhs_.resize(rows_, std::vector<double>(cols_, INF));
        map_ = sensed_map;

        rhs_[goal_.first][goal_.second] = 0.0;

        auto keyVal = calculate_key(goal_);
        queue_.push(Element(goal_, keyVal.first, keyVal.second));
    }

    // plan 函数
    std::vector<std::pair<int,int>> plan(std::pair<int,int> current_pos) {
        start_ = current_pos;
        compute_shortest_path();

        std::vector<std::pair<int,int>> path;
        auto next_node = start_;
        std::set<std::pair<int,int>> visited;

        while(next_node != goal_) {
            path.push_back(next_node);
            visited.insert(next_node);

            // 收集邻居
            std::vector<std::pair<int,int>> next_options;
            for(const auto& n: succ(next_node)) {
                if(map_[n.first][n.second] != 1
                   && g_[n.first][n.second] < INF
                   && visited.find(n) == visited.end()) 
                {
                    next_options.push_back(n);
                }
            }

            if(next_options.empty()) {
                // 没有更优邻居时，备选一个
                std::vector<std::pair<int,int>> backup_options;
                for(const auto& n: succ(next_node)) {
                    if(map_[n.first][n.second] != 1 
                       && visited.find(n) == visited.end()) 
                    {
                        backup_options.push_back(n);
                    }
                }
                if(!backup_options.empty()) {
                    // 随机选一个
                    static std::mt19937 gen(std::random_device{}());
                    std::uniform_int_distribution<> dist(0, (int)backup_options.size()-1);
                    next_node = backup_options[dist(gen)];
                    continue;
                }
                break;
            }

            double best_cost = std::numeric_limits<double>::infinity();
            std::pair<int,int> best_node = next_node;
            for(const auto& candidate: next_options) {
                double cost = g_[candidate.first][candidate.second] 
                              + h_estimate(candidate, goal_);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_node = candidate;
                }
            }
            next_node = best_node;
        }

        if(next_node == goal_) {
            path.push_back(goal_);
        }
        return path;
    }

    void reset_partial() {
        k_m_ = 0.0;
        for(int i = 0; i < rows_; ++i) {
            for(int j = 0; j < cols_; ++j) {
                g_[i][j]   = INF;
                rhs_[i][j] = INF;
            }
        }
        rhs_[goal_.first][goal_.second] = 0.0;

        // 清空队列后重新推入目标节点
        while(!queue_.empty()) {
            queue_.pop();
        }
        auto keyVal = calculate_key(goal_);
        queue_.push(Element(goal_, keyVal.first, keyVal.second));
    }

    bool is_path_valid(const std::vector<std::pair<int,int>>& path) {
        for(const auto& node: path) {
            if(map_[node.first][node.second] == 1) {
                return false;
            }
        }
        return true;
    }

    // 可以在Python端直接访问或修改 sensed_map
    const std::vector<std::vector<int>>& get_sensed_map() const {
        return map_;
    }

    void set_sensed_map(const std::vector<std::vector<int>>& new_map) {
        map_ = new_map;
    }

    // 供 global_planning 调用
    const std::pair<int,int>& get_goal() const { return goal_; }
    const std::pair<int,int>& get_start() const { return start_; }

    // 在地图发生变化后，重新更新受影响节点
    void update_sensed_map() {
        for(int i = 0; i < rows_; ++i) {
            for(int j = 0; j < cols_; ++j) {
                if(map_[i][j] == 1) {
                    auto neighbors = succ({i,j});
                    for(const auto& n: neighbors) {
                        update_vertex(n);
                    }
                }
            }
        }
    }

private:
    std::pair<double,double> calculate_key(const std::pair<int,int>& node) const {
        double g_rhs_min = std::min(g_[node.first][node.second],
                                    rhs_[node.first][node.second]);
        double key1 = g_rhs_min + h_estimate(start_, node) + k_m_;
        double key2 = g_rhs_min;
        return {key1, key2};
    }

    void update_vertex(const std::pair<int,int>& u) {
        if(u != goal_) {
            double min_rhs = INF;
            for(const auto& s: succ(u)) {
                if(map_[s.first][s.second] != 1) {
                    double tmp = cost(u, s) + g_[s.first][s.second];
                    if(tmp < min_rhs) {
                        min_rhs = tmp;
                    }
                }
            }
            rhs_[u.first][u.second] = min_rhs;
        }
        // 从队列中去掉 u
        std::priority_queue<Element> new_queue;
        while(!queue_.empty()) {
            auto top_el = queue_.top();
            queue_.pop();
            if(top_el.key != u) {
                new_queue.push(top_el);
            }
        }
        queue_ = std::move(new_queue);

        if(g_[u.first][u.second] != rhs_[u.first][u.second]) {
            auto keyVal = calculate_key(u);
            queue_.push(Element(u, keyVal.first, keyVal.second));
        }
    }

    void compute_shortest_path() {
        while(!queue_.empty()) {
            auto top_el = queue_.top();
            auto start_key = calculate_key(start_);
            bool cond1 = false;
            if(std::make_pair(top_el.value1, top_el.value2)
               < std::make_pair(start_key.first, start_key.second)) {
                cond1 = true;
            }
            bool cond2 = (rhs_[start_.first][start_.second] != g_[start_.first][start_.second]);
            if(!cond1 && !cond2) {
                break;
            }

            queue_.pop();
            auto u = top_el.key;
            if(g_[u.first][u.second] > rhs_[u.first][u.second]) {
                g_[u.first][u.second] = rhs_[u.first][u.second];
                auto neighbors = succ(u);
                for(const auto& s: neighbors) {
                    update_vertex(s);
                }
            } else {
                g_[u.first][u.second] = INF;
                auto neighbors = succ(u);
                neighbors.push_back(u);
                for(const auto& s: neighbors) {
                    update_vertex(s);
                }
            }
        }
    }

    std::vector<std::pair<int,int>> succ(const std::pair<int,int>& u) const {
        static const int dd[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        std::vector<std::pair<int,int>> result;
        for(int i = 0; i < 4; i++){
            int nr = u.first + dd[i][0];
            int nc = u.second + dd[i][1];
            if(nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_){
                result.emplace_back(nr,nc);
            }
        }
        return result;
    }

    double h_estimate(const std::pair<int,int>& s1, const std::pair<int,int>& s2) const {
        return std::abs(s1.first - s2.first)
             + std::abs(s1.second - s2.second);
    }

    double cost(const std::pair<int,int>& u1, const std::pair<int,int>& u2) const {
        return h_estimate(u1, u2);
    }

    int rows_;
    int cols_;
    double k_m_;

    std::pair<int,int> start_;
    std::pair<int,int> goal_;

    std::vector<std::vector<double>> g_;
    std::vector<std::vector<double>> rhs_;
    std::vector<std::vector<int>>    map_;

    std::priority_queue<Element> queue_;
};

// global_planning 函数
std::pair<int, std::vector<std::pair<int,int>>>
global_planning(DLitePlanner& agent,
                std::pair<int,int> current_pos,
                std::vector<std::pair<int,int>> paths,
                const std::vector<std::vector<int>>& explored_map)
{
    int action = 0;

    // 一些辅助量
    auto goal = agent.get_goal();
    auto start = agent.get_start();

    // 如果原先的paths为空
    if(paths.empty()) {
        agent.reset_partial();
        paths = agent.plan(current_pos);
        if(current_pos == goal) {
            action = 0;
        } else {
            if(paths.size() <= 1) {
                action = 0;
                paths.clear();
            } else {
                auto next_step = paths[1];
                int dx = next_step.first - current_pos.first;
                int dy = next_step.second - current_pos.second;
                action = get_action_from_displacement(dx, dy);
                paths.erase(paths.begin());
            }
        }
    }
    else if(paths.size() > 1) {
        // 如果已存在路径，但地图上发现障碍，需要重新规划
        for(size_t i = 0; i < paths.size(); i++){
            auto next_path = paths[i];
            if(explored_map[next_path.first][next_path.second] == 1) {
                agent.reset_partial();
                paths = agent.plan(current_pos);
                break;
            }
        }
        if(paths.size() <= 1) {
            action = 0;
            paths.clear();
        } else {
            auto next_step = paths[1];
            int dx = next_step.first - current_pos.first;
            int dy = next_step.second - current_pos.second;
            action = get_action_from_displacement(dx, dy);
            paths.erase(paths.begin());
        }
    }
    else {
        // 当只剩1个节点时
        if(current_pos == goal) {
            action = 0;
        } else {
            agent.reset_partial();
            paths = agent.plan(current_pos);
            if(paths.size() <= 1) {
                action = 0;
                paths.clear();
            } else {
                auto next_step = paths[1];
                int dx = next_step.first - current_pos.first;
                int dy = next_step.second - current_pos.second;
                action = get_action_from_displacement(dx, dy);
                paths.erase(paths.begin());
            }
        }
    }

    return std::make_pair(action, paths);
}

// 使用pybind11进行包装
PYBIND11_MODULE(dliteplanner, m) {
    py::class_<DLitePlanner>(m, "DLitePlanner")
        .def(py::init<const std::vector<std::vector<int>>&,
                      std::pair<int,int>,
                      std::pair<int,int>>(),
             py::arg("sensed_map"),
             py::arg("start"),
             py::arg("goal"))
        .def("plan", &DLitePlanner::plan)
        .def("reset_partial", &DLitePlanner::reset_partial)
        .def("is_path_valid", &DLitePlanner::is_path_valid)
        .def("update_sensed_map", &DLitePlanner::update_sensed_map)
        .def("get_sensed_map", &DLitePlanner::get_sensed_map)
        .def("set_sensed_map", &DLitePlanner::set_sensed_map);

    m.def("global_planning", &global_planning,
          py::arg("agent"),
          py::arg("current_pos"),
          py::arg("paths"),
          py::arg("explored_map"));
}
