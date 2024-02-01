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
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def build_reward_matrix(edges: pd.DataFrame, N_nodes: int, target: int) -> np.ndarray:
    R_mtx = np.zeros((N_nodes, N_nodes))  # 奖励矩阵
    R_mtx[edges["source"], edges["target"]] = 1 / edges["tt"]  # NOTE: 这里取逆，是因为tt越小奖励越大
    R_mtx[target, target] = 10
    return R_mtx
    

if __name__ == "__main__":
    edges = pd.read_csv(f"{BASE_DIR}/file/edges.csv")
    edges[["source", "target"]] -= 1
    
    nodes = np.unique(edges[["source", "target"]])
    N_nodes = len(nodes)
    
    target = 61
    
    # ---- 参数构建 ---------------------------------------------------------------------------------
    
    R_mtx = build_reward_matrix(edges, N_nodes, target)
    Q_mtx = np.zeros((N_nodes, N_nodes))
    
    # TODO: 继续完成剩余部分
    
    
    
    