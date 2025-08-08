# 问题二：多通道LED光源优化解决方案

## 问题描述

多通道光源通过多个独立的发光通道（LED芯片）组合，从而灵活地合成多种光谱功率分布（SPD）。通过精确调节各通道的驱动权重，可以对合成光谱的形状、色温（CCT）、显色性（Rg/Rf）及生理节律效应（mel-DER）等关键特性进行动态控制。

### 给定条件
- **5个LED通道**：蓝光、绿光、红光、暖白光(WW, ~3000K)、冷白光(CW, ~6500K)
- **合成原理**：多通道光源的总光谱是各个通道光谱的加权线性叠加

## 核心数学公式

### 1. 光谱合成公式

多通道LED光源的合成光谱功率分布为：

$$S_{\text{合成}}(\lambda) = \sum_{i=1}^{5} w_i \times S_i(\lambda)$$

其中：
- $S_{\text{合成}}(\lambda)$ - 合成光谱在波长$\lambda$处的功率
- $w_i$ - 第$i$个通道的权重系数
- $S_i(\lambda)$ - 第$i$个通道在波长$\lambda$处的光谱功率
- 通道顺序：[蓝光, 绿光, 红光, 暖白光, 冷白光]

### 2. 权重归一化

为确保合成光谱的物理意义，权重需要归一化：

$$w_{i,\text{norm}} = \frac{w_i}{\sum_{j=1}^{5} w_j}$$

### 3. 关键参数计算

基于合成光谱计算以下参数：

#### 3.1 CIE XYZ三刺激值
$$X = k \times \int S_{\text{合成}}(\lambda) \times \bar{x}(\lambda) d\lambda$$
$$Y = k \times \int S_{\text{合成}}(\lambda) \times \bar{y}(\lambda) d\lambda$$
$$Z = k \times \int S_{\text{合成}}(\lambda) \times \bar{z}(\lambda) d\lambda$$

其中：
- $k = \frac{100}{\int S_{\text{合成}}(\lambda) \times \bar{y}(\lambda) d\lambda}$ - 归一化常数
- $\bar{x}(\lambda), \bar{y}(\lambda), \bar{z}(\lambda)$ - CIE 1931标准观察者颜色匹配函数

#### 3.2 相关色温(CCT)计算
通过CIE 1960 UCS色度坐标：
$$u = \frac{4X}{X + 15Y + 3Z}$$
$$v = \frac{6Y}{X + 15Y + 3Z}$$

CCT通过最小化距离黑体轨迹的欧氏距离求得：
$$\text{CCT} = \arg\min_T \sqrt{(u - u_T)^2 + (v - v_T)^2}$$

其中$u_T, v_T$是温度$T$对应的黑体轨迹坐标。

#### 3.3 颜色渲染指数
- **保真度指数($R_f$)**：使用ANSI/IES TM-30-18标准计算
- **色域指数($R_g$)**：评估光源相对于参考光源的色域面积比

#### 3.4 褪黑素日光照度比(mel-DER)
$$\text{mel-DER} = \frac{E_{\text{mel,test}} / E_{\text{phot,test}}}{E_{\text{mel,D65}} / E_{\text{phot,D65}}}$$

其中：
- $E_{\text{mel}} = \int S(\lambda) \times s_{\text{mel}}(\lambda) d\lambda$ - 褪黑素辐照度
- $E_{\text{phot}} = \int S(\lambda) \times V(\lambda) d\lambda \times 683$ - 明视觉照度
- $s_{\text{mel}}(\lambda)$ - CIE S026褪黑素效应函数
- $V(\lambda)$ - 明视觉光效函数

## 优化策略

### 场景一：日间照明模式

**目标函数**：
$$\min: f_{\text{obj}} = \begin{cases}
-(R_f + 20) & \text{if } R_f \geq 95 \text{ (强奖励超高显色性)} \\
-(R_f + 5) & \text{if } 90 \leq R_f < 95 \text{ (奖励高显色性)} \\
-R_f & \text{otherwise}
\end{cases}$$

**约束条件**：
- $5500\text{K} \leq \text{CCT} \leq 6500\text{K}$ (模拟正午日光)
- $95 \leq R_g \leq 105$ (色域指数范围)  
- $R_f \geq 88$ (基本颜色还原要求)
- $R_f \geq 95$ (目标高显色性要求)

**优化后的带惩罚项目标函数**：
$$f_{\text{day}} = f_{\text{obj}} + P_{\text{CCT}} + P_{R_g} + P_{R_f} + P_{\text{quality}}$$

其中惩罚项（进一步优化）：
$$P_{\text{CCT}} = \begin{cases}
800 \times \frac{|\text{CCT} - 6000|}{1000} & \text{if CCT} \notin [5500, 6500] \\
0 & \text{otherwise}
\end{cases}$$

$$P_{R_g} = 100 \times \max(0, 95-R_g, R_g-105)$$

$$P_{R_f} = \begin{cases}
1000 \times (88-R_f) & \text{if } R_f < 88 \\
0 & \text{otherwise}
\end{cases}$$

$$P_{\text{quality}} = \begin{cases}
200 \times (95-R_f) & \text{if } R_f < 95 \text{ (强鼓励高显色性)} \\
0 & \text{otherwise}
\end{cases}$$

### 场景二：夜间助眠模式

**目标函数**：
$$\min: \text{mel-DER} \quad \text{(最小化生理节律干扰)}$$

**约束条件**：
- $2500\text{K} \leq \text{CCT} \leq 3500\text{K}$ (营造温馨低色温环境)
- $R_f \geq 80$ (保证基本颜色分辨能力)

**带惩罚项的目标函数**：
$$f_{\text{night}} = \text{mel-DER} + P_{\text{CCT}} + P_{R_f}$$

其中惩罚项：
$$P_{\text{CCT}} = \begin{cases}
100 \times \frac{|\text{CCT} - 3000|}{1000} & \text{if CCT} \notin [2500, 3500] \\
0 & \text{otherwise}
\end{cases}$$

$$P_{R_f} = \begin{cases}
100 \times (80-R_f) & \text{if } R_f < 80 \\
0 & \text{otherwise}
\end{cases}$$

## 遗传算法实现

### 算法参数
- **种群大小**：100
- **进化代数**：200
- **变异率**：0.1
- **交叉率**：0.7

### 个体编码
每个个体用5维实数向量表示：$\mathbf{w} = [w_{\text{blue}}, w_{\text{green}}, w_{\text{red}}, w_{\text{warm}}, w_{\text{cold}}]$

### 关键操作

#### 1. 适应度计算
$$\text{fitness} = \frac{1}{1 + \text{objective\_value}}$$

#### 2. 选择策略
轮盘赌选择：基于适应度比例进行个体选择

#### 3. 交叉操作
单点交叉：随机选择交叉点，交换父代基因片段

#### 4. 变异操作
高斯变异：对每个基因添加正态分布随机噪声
$$w_{i,\text{new}} = w_i + \mathcal{N}(0, 0.1)$$
$$w_{i,\text{new}} = \text{clip}(w_{i,\text{new}}, 0, 1)$$

## 算法流程

$$
\begin{align}
&\text{1. 初始化：} \\
&\quad \text{随机生成初始种群（100个个体）} \\
&\quad \text{每个个体为5维权重向量} \mathbf{w}^{(0)}_i \sim \mathcal{U}(0,1)^5 \\
\\
&\text{2. 迭代优化（200代）：} \\
&\quad \text{For } t = 1 \text{ to } 200: \\
&\quad\quad \text{a) 权重归一化: } \tilde{\mathbf{w}}_i = \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|_1} \\
&\quad\quad \text{b) 合成光谱计算: } S(\lambda) = \sum_{j=1}^5 \tilde{w}_{i,j} S_j(\lambda) \\
&\quad\quad \text{c) 参数计算（CCT, } R_f, R_g, \text{mel-DER）} \\
&\quad\quad \text{d) 目标函数评估: } f_i = f(\tilde{\mathbf{w}}_i) \\
&\quad\quad \text{e) 适应度计算: } \phi_i = \frac{1}{1+f_i} \\
&\quad\quad \text{f) 选择操作} \\
&\quad\quad \text{g) 交叉操作} \\
&\quad\quad \text{h) 变异操作} \\
&\quad\quad \text{i) 更新种群} \\
\\
&\text{3. 输出结果：} \\
&\quad \text{最优权重组合} \mathbf{w}^* \\
&\quad \text{合成光谱参数} \\
&\quad \text{优化历史曲线}
\end{align}
$$

## 结果分析与性能评估

### 优化算法性能指标

#### 日间照明模式优化结果
- **CCT范围**：5500-6500K (✓ 目标达成)
- **$R_f$水平**：92-95+ (优秀显色性能)
- **$R_g$范围**：95-105 (✓ 色域达标)
- **mel-DER**：0.6-0.8 (中等生理刺激，日间可接受)

#### 夜间助眠模式优化结果  
- **CCT范围**：2500-3500K (✓ 目标达成)
- **$R_f$水平**：85-90+ (良好显色性能)
- **mel-DER**：0.3-0.4 (低生理刺激，助眠效果良好)

### mel-DER水平标准对比

$$\text{mel-DER相对评估} = \begin{cases}
\text{极优 (≤0.1)} & \text{极低生理刺激，优秀助眠效果} \\
\text{优秀 (0.1-0.3)} & \text{低生理刺激，良好助眠效果} \\
\text{良好 (0.3-0.6)} & \text{中等生理刺激，可接受助眠效果} \\
\text{需改进 (>0.6)} & \text{较高生理刺激，建议进一步优化}
\end{cases}$$

**参考基准**：
- 标准日光D65：mel-DER = 1.0 (基准值)
- 传统白炽灯(3000K)：mel-DER ≈ 0.3-0.4
- 优化后夜间模式：mel-DER ≈ 0.4 (相比D65降低60%)

### 性能指标对比表

| 模式 | CCT(K) | $R_f$ | $R_g$ | mel-DER | 性能评级 | 应用场景 |
|------|--------|-------|-------|---------|----------|----------|
| 日间照明 | 5600-6400 | 92-95 | 95-105 | 0.6-0.8 | ★★★★★ | 办公、阅读、精细作业 |
| 夜间助眠 | 2600-3400 | 85-90 | 不限 | 0.3-0.4 | ★★★★☆ | 卧室、休闲、睡前照明 |

### 权重分布特征分析

#### 日间模式典型权重分布
$$\mathbf{w}_{\text{day}} \approx [0.17, 0.17, 0.25, 0.00, 0.41]$$
- **冷白光主导**(41%)：提供充足的短波蓝光成分
- **红光辅助**(25%)：平衡光谱，优化显色性
- **蓝绿均衡**(17% each)：补充光谱连续性
- **暖白最少**(0%)：避免色温偏低

#### 夜间模式典型权重分布  
$$\mathbf{w}_{\text{night}} \approx [0.05, 0.06, 0.34, 0.36, 0.19]$$
- **暖白光主导**(36%)：提供低色温基础
- **红光强化**(34%)：增强长波成分，降低mel-DER
- **冷白适量**(19%)：维持基本显色能力
- **蓝绿最少**(5-6%)：最小化生理节律干扰

### 算法收敛性能

$$\text{收敛效率} = \frac{\text{目标函数改进幅度}}{\text{计算代数}} \times 100\%$$

- **日间模式**：通常在30-50代内收敛到$R_f > 92$
- **夜间模式**：通常在20-30代内收敛到mel-DER < 0.4
- **计算复杂度**：$O(N \times G \times M)$，其中$N$为种群大小，$G$为代数，$M$为光谱点数

### 权重分布特征
- **日间模式**：冷白光和蓝光权重较高，提供充足的短波成分
- **夜间模式**：暖白光和红光权重较高，减少蓝光成分

## 技术要点

### 1. 数值积分
使用梯形积分法计算光谱相关积分：
$$\int f(\lambda) d\lambda \approx \sum_{i=1}^{n-1} \frac{f(\lambda_i) + f(\lambda_{i+1})}{2} \cdot (\lambda_{i+1} - \lambda_i)$$

```python
∫ f(λ) dλ ≈ np.trapz(f_values, wavelengths)
```

### 2. 插值处理
将不同波长范围的SPD数据插值到统一网格：
$$S_{\text{interp}}(\lambda) = \text{LinearInterp}([\lambda_1, \lambda_2, ..., \lambda_n], [S_1, S_2, ..., S_n])(\lambda)$$

```python
f_interp = interp1d(wavelengths, spd_values, kind='linear')
```

### 3. 异常处理
- 零光谱检测和处理
- 数值计算异常的fallback机制
- 约束违反的惩罚机制

## 实际应用意义

1. **智能照明系统**：根据时间和场景自动调节光谱
2. **健康照明**：基于生理节律优化的照明方案
3. **高质量显色**：满足专业照明的颜色还原需求
4. **节能优化**：在满足性能要求下优化能耗分配

## 扩展方向

1. **多目标优化**：同时优化多个性能指标
2. **动态优化**：考虑时间变化的光谱需求
3. **个性化优化**：基于用户偏好的定制化光谱
4. **硬件约束**：考虑LED驱动电路的实际限制

---

*本文档基于2025年华数杯全国大学生数学建模竞赛C题的解决方案*
