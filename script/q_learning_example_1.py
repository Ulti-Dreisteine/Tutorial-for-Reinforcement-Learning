# -*- coding: utf-8 -*-
"""
Created on 2024/01/26 19:57:09

@File -> q_learning_example.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Q学习案例1——寻找房间出口
"""

import numpy as np
import random

states = np.arange(6)
actions = np.arange(6)

# ---- 设置奖励矩阵R -----------------------------------------------------------------------------

# 奖励矩阵，行：状态、列：动作、元素：奖励值
R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100],
])

# 状态邻接矩阵：row -> col
A = np.array([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1],
])

# ---- Q-Learning ------------------------------------------------------------------------------

gamma = 0.8

# 初始化Q矩阵，行：状态、列：动作、元素：Q值
Q = np.zeros((len(states), len(actions)))

# 目标节点
s_end = 5

# 总迭代数, 起始迭代数
iter, i = 1000, 0

while True:
    # 开始Episode: 随机选择一个状态
    s = random.choice(states)
    
    while True:
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
        cand_actions_next = cand_states_next
        max_q = np.max(Q[s_next, cand_actions_next])
        
        # 更新Q元素
        Q[s, a] = r + gamma * max_q
        
        if s_next == s_end:
            break
        else:
            s = s_next
    
    i += 1
    
    if i == iter:
        break

Q = np.round(Q, 2)