# -*- coding: utf-8 -*-
"""
Created on 2024/02/01 09:25:23

@File -> q_learning_exp_2.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Q学习案例2：路径寻优. 寻找在从一个有向图上任意节点出发，前往目标点的最优路由方案。
"""

import seaborn as sns
import pandas as pd
import numpy as np
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def build_reward_matrix(edges: pd.DataFrame, N_nodes: int, target: int) -> np.ndarray:
    """
    构建奖励矩阵
    """
    
    R = np.zeros((N_nodes, N_nodes))  # 奖励矩阵
    R[edges["source"], edges["target"]] = 1 / edges["tt"]  # NOTE: 这里取逆，是因为tt越小奖励越大
    R[target, target] = 10
    return R


def build_adj_matrix(edges: pd.DataFrame, N_nodes: int, target: int) -> np.ndarray:
    """
    构建状态间的邻接矩阵
    """
    
    A = np.zeros((N_nodes, N_nodes))
    A[edges["source"], edges["target"]] = 1
    A[target, target] = 1
    
    return A


if __name__ == "__main__":
    
    # ---- 数据读取 ---------------------------------------------------------------------------------
    
    # 读取边表
    edges = pd.read_csv(f"{BASE_DIR}/file/edges.csv")
    edges[["source", "target"]] -= 1
    
    # 节点集
    nodes = np.sort(np.unique(edges[["source", "target"]]))
    
    target = 60
    
    # ---- 参数构建 ---------------------------------------------------------------------------------
    
    states = set(nodes)
    actions = set(nodes)
    states_end = set([59, 60, 61, 62, 64, 41, 42, 43])  # 所有任务区的输入节点集
    s_end = 60
    N_states = len(states)
    N_actions = len(actions)
    
    R = build_reward_matrix(edges, N_states, target)
    Q = np.zeros((N_states, N_actions))
    A = build_adj_matrix(edges, N_states, s_end)
    
    # ---- Q学习 ------------------------------------------------------------------------------------
    
    gamma = 0.8
    
    # 总迭代数, 起始迭代数
    iter, i = 1000, 0
    
    while True:
        # 开始Episode: 随机选择一个非终点的起始状态
        s = random.choice(list(states.difference(states_end)))
        
        while True:
            # TODO: 对邻接状态集为空的处理方式
            
            # 查看状态s的可达状态，获得可选动作
            cand_states = np.argwhere(A[s, :] > 0).flatten()
            
            # 随机选择一个动作
            cand_actions = cand_states
            a = random.choice(cand_actions)
            
            # 下一状态和奖励
            r = R[s, a]
            s_next = a
            
            # 查看状态s_next的可达状态，获得可选动作
            cand_states_next = np.argwhere(A[s_next, :] > 0).flatten()
            
            # 求解max_{a_next} Q(s_next, a_next)
            if len(cand_actions_next) > 0:
                cand_actions_next = cand_states_next
                max_q = np.max(Q[s_next, cand_actions_next])
            
            # 更新Q元素
            Q[s, a] = r + gamma * max_q
            
            if s_next in states_end:
                break
            else:
                s = s_next
        
        i += 1
        
        if i == iter:
            break

    Q = np.round(Q, 2)
    
    
    
    
    
    