### Q-Learning

Q-Learning是强化学习方法中的一种，适合初学者入门。  
适用条件：状态（state）和动作（action）空间离散且数量少。


#### 两张重要的表

**奖励-动作表（R表）**，表示为 ${\bold R}_{N_{\rm s} \times N_{\rm a}}$，其中 $R(s,a)$ 表示在状态 $s$ 下采取动作 $a$ 所得（期望）奖励值：

$$
\begin{array}{c}
                    &   &           &  &\text{action}      &       &       & \\
                    &   &1          &2          &3          &4      &\cdots &N_{\rm a} \\ 
                    &1  &R_{1,1}    &R_{1,2}    &R_{1,3}    &R(1,4)&\cdots &R(1, N_{\rm a}) \\
     \text{state}   &2  &R_{2,1}    &R_{2,2}    &R_{2,3}    &R(2,4)&\cdots &R(2, N_{\rm a}) \\
                    &\vdots &\vdots &\vdots &\vdots &\vdots &\ddots &\vdots \\
                    &N_{\rm s} &R(N_{\rm s},1) &R(N_{\rm s},2) &R(N_{\rm s},3) &R(N_{\rm s},4) &\cdots &R(N_{\rm s},N_{\rm a}) \\
\end{array}
$$

**状态-动作表（Q表）**，表示为 ${\bold Q}_{N_{\rm s} \times N_{\rm a}}$，其中 $Q(s,a)$ 表示在状态 $s$ 下采取动作 $a$ 的Q值：

$$
\begin{array}{c}
                    &   &           &  &\text{action}      &       &       & \\
                    &   &1          &2          &3          &4      &\cdots &N_{\rm a} \\ 
                    &1  &Q(1,1)    &Q(1,2)    &Q(1,3)    &Q(1,4)&\cdots &Q(1, N_{\rm a}) \\
     \text{state}   &2  &Q(2,1)    &Q(2,2)    &Q(2,3)    &Q(2,4)&\cdots &Q(2, N_{\rm a}) \\
                    &\vdots &\vdots &\vdots &\vdots &\vdots &\ddots &\vdots \\
                    &N_{\rm s} &Q(N_{\rm s},1) &Q(N_{\rm s},2) &Q(N_{\rm s},3) &Q(N_{\rm s},4) &\cdots &Q(N_{\rm s},N_{\rm a}) \\
\end{array}
$$

#### Q-Learning过程

先初始化Q表，将所有的 $Q(s,a)$ 都设为0，然后随机选择初始状态和动作，通过贝尔曼方程迭代不断更新表格中的值函数，直至算法收敛，得到稳定的Q表。

Q-Learning迭代过程为如下的**贝尔曼方程**（Bellman Equation）：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[R(s, a) + \gamma \max_{a'} Q(s',a') - Q(s, a)\right] \tag{1}
$$

式中，$R$ 是在当前状态 $s$ 执行动作 $a$ 后的回报；$\max_{a'} Q(s',a')$ 是指在下一状态 $s'$ 取得的最大Q值；$\alpha$ 为学习率，$\gamma$ 为折扣因子。

如果取 $\alpha = 1$，则有

$$
Q(s,a) \leftarrow R(s, a) + \gamma \max_{a'} Q(s',a') \tag{2}
$$
