# -*- coding: utf-8 -*-
"""
Created on 2024/02/01 14:43:34

@File -> q_learning_exp_2_routes_on_graph.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Q学习案例2：在一个有向图上，寻找从任意节点出发，前往设定目标点的最优（省时的）路由方案。
"""

from typing import Tuple
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def read_edge_table() -> Tuple[list, list]:
    edges = pd.read_csv(f"{BASE_DIR}/file/edges.csv")
    
    # 将节点编号转为从0开始，便于后续R和Q矩阵的查询
    edges[["source", "target"]] -= 1
    
    # 节点集和边集提取
    nodes = np.sort(np.unique(edges[["source", "target"]])).tolist()
    edges = edges.values[:, :-1].tolist()
    
    return nodes, edges


def build_reward_matrix(g: nx.DiGraph, states_e: set, state_target: int) -> np.ndarray:
    """
    构建奖励矩阵
    """
    
    # 初始化
    R = np.zeros((len(g.nodes), len(g.nodes)))
    
    # 图中所有有向边的奖励为对应tt值的倒数
    for (state_s, state_t) in g.edges:
        R[state_s, state_t] = 1 / g.edges[state_s, state_t]["tt"]
    
    # 目标点上加入自环
    R[state_target, state_target] = 3
    
    return R
    

if __name__ == "__main__":
    
    # ---- 数据读取 ---------------------------------------------------------------------------------
    
    nodes, edges = read_edge_table()
    
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges, weight="tt")
    
    # ---- 参数准备 ---------------------------------------------------------------------------------
    
    states = set(g.nodes)
    actions = set(g.nodes)
    N_states = len(states)
    N_actions = len(actions)
    
    # 系统的终点集合
    states_e = set([p for p in g.nodes if not list(g.successors(p))])
    
    # 系统终点状态加入自环
    for state in states_e:
        g.add_edge(state, state, tt=1)
    
    # 目标状态
    state_target = 60
    
    # 建立奖励函数矩阵
    R = build_reward_matrix(g, states_e, state_target)
    
    # ---- Q学习 -----------------------------------------------------------------------------------
    
    # 衰减系数
    gamma = 0.8
    
    # 总迭代数、起始计数
    iters, i = 1, 0
    
    Q = np.zeros((N_states, N_actions))
    
    while True:
        # 开始Episode: 随机选择一个起始状态
        s = random.choice(list(states))
        
        while True:
            # 查看状态s的可达状态，获得可选动作
            cand_states = list(g.successors(s))
            
            # 随机选择一个动作
            cand_actions = cand_states
            a = random.choice(cand_actions)
            
            # 下一状态和奖励
            r = R[s, a]
            s_next = a
            
            # 查看状态s_next的可达状态，获得可选动作
            cand_states_next = list(g.successors(s_next))
            
            # 求解max_{a_next} Q(s_next, a_next)
            if len(cand_states_next) > 0:
                cand_actions_next = cand_states_next
                max_q = np.max(Q[s_next, cand_actions_next])
            
            # 更新Q元素
            Q[s, a] = r + gamma * max_q
            
            if s_next in states_e:
                break
            else:
                s = s_next
        
        i += 1
        
        if i == iters:
            break

    Q = np.round(Q, 2)
    
    
    