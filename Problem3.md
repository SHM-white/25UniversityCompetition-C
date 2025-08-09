# 问题三：全天候太阳光模拟LED控制策略

## 问题描述

人类的生理和心理健康与一天中自然光的变化息息相关。需要设计一个控制策略，使用问题二中的五通道LED系统来模拟从早晨8:30到傍晚19:30的太阳光谱数据，实现相似的节律效果。

## 核心思路与数学模型

### 1. 问题分析

#### 1.1 目标
- 使用5通道LED系统（蓝光、绿光、红光、暖白光WW、冷白光CW）
- 模拟给定时间序列的太阳光谱（8:30-19:30）
- 保持相似的生理节律效应

#### 1.2 关键挑战
- 太阳光谱的连续变化特性
- 不同时间点光谱特征差异显著
- 5通道LED的光谱合成能力有限

### 2. 数学建模方法

#### 2.1 光谱匹配优化模型

对于每个时间点$t$，建立优化模型：

**目标函数：最小化光谱差异**
$$\min_{w_1,w_2,w_3,w_4,w_5} \sum_{\lambda=380}^{780} \left[S_{\text{sun}}(\lambda,t) - \sum_{i=1}^{5} w_i S_i(\lambda)\right]^2$$

其中：
- $S_{\text{sun}}(\lambda,t)$ - 时间$t$的太阳光谱
- $S_i(\lambda)$ - 第$i$个LED通道的光谱
- $w_i$ - 第$i$个通道的权重

**约束条件：**
- $w_i \geq 0$ （权重非负）
- $\sum_{i=1}^{5} w_i = 1$ （权重归一化）

#### 2.2 多目标优化策略

结合关键参数匹配：

$$\min \left[ \alpha \cdot f_{\text{spectrum}} + \beta \cdot f_{\text{CCT}} + \gamma \cdot f_{\text{Duv}} + \delta \cdot f_{\text{mel-DER}} \right]$$

其中：
- $f_{\text{spectrum}} = \sum_{\lambda} [S_{\text{sun}}(\lambda,t) - S_{\text{合成}}(\lambda)]^2$ - 光谱匹配误差
- $f_{\text{CCT}} = \frac{|CCT_{\text{sun}}(t) - CCT_{\text{合成}}|}{CCT_{\text{sun}}(t)}$ - 色温相对误差  
- $f_{\text{Duv}} = \frac{|Duv_{\text{sun}}(t) - Duv_{\text{合成}}|}{|Duv_{\text{sun}}(t)|}$ - Duv相对误差
- $f_{\text{mel-DER}} = \frac{|mel\text{-}DER_{\text{sun}}(t) - mel\text{-}DER_{\text{合成}}|}{mel\text{-}DER_{\text{sun}}(t)}$ - 节律效应相对误差

权重系数$\alpha, \beta, \gamma, \delta$可根据重要性调整。

**关键改进：增加Duv匹配误差**
- Duv（距离普朗克轨迹的距离）反映了色坐标的偏差程度
- 对于精确的颜色重现，Duv匹配至关重要
- 采用相对误差形式，与CCT和mel-DER误差形式保持一致

### 3. 基于前两问的核心公式应用

#### 3.1 合成光谱计算（来自问题二）

$$S_{\text{合成}}(\lambda) = \sum_{i=1}^{5} w_i \times S_i(\lambda)$$

#### 3.2 关键参数计算（来自问题一）

**相关色温CCT：**
1. 计算CIE XYZ三刺激值：
   $$X = k \sum_{\lambda} S_{\text{合成}}(\lambda)\,\bar{x}(\lambda)\\
   Y = k \sum_{\lambda} S_{\text{合成}}(\lambda)\,\bar{y}(\lambda)\\  
   Z = k \sum_{\lambda} S_{\text{合成}}(\lambda)\,\bar{z}(\lambda)$$

2. 转换到CIE 1960 UCS：
   $$u = \frac{4X}{X+15Y+3Z}, \quad v = \frac{6Y}{X+15Y+3Z}$$

3. 通过最小化距离黑体轨迹求CCT

**距离普朗克轨迹的距离Duv：**
$$Duv = \text{sign} \times \sqrt{(u - u_t)^2 + (v - v_t)^2}$$

其中：
- $(u, v)$ - 光源在CIE 1960 UCS色度图中的坐标
- $(u_t, v_t)$ - 对应CCT在普朗克轨迹上的坐标点
- $\text{sign}$ - 符号，表示偏离方向

**褪黑素日光照度比mel-DER：**
$$mel\text{-}DER = \frac{E_{\text{mel}}}{E_V}$$

其中：
- $E_{\text{mel}}$ - 褪黑素照度
- $E_V$ - 明视觉照度

### 4. 算法实现策略

#### 4.1 时间序列处理
1. 读取太阳光谱时间序列数据
2. 对每个时间点计算目标参数（CCT, mel-DER等）
3. 建立时间-参数映射关系

#### 4.2 优化算法选择
**遗传算法（继承问题二代码）：**
- 个体编码：5维权重向量$[w_1, w_2, w_3, w_4, w_5]$
- 适应度函数：基于光谱匹配误差
- 约束处理：权重归一化

**序列最小二乘法：**
对于线性约束的二次优化问题，可使用SLSQP算法

#### 4.3 时间插值策略
对于离散时间点之间的连续控制：
- 线性插值权重
- 样条插值实现平滑过渡

### 5. 代码实现框架

```python
class SolarSpectrumMimicry:
    def __init__(self, led_channels, solar_data):
        self.led_channels = led_channels  # 5通道LED数据
        self.solar_data = solar_data      # 太阳光谱时间序列
        self.calculator = SPDCalculator() # 复用问题一的计算器
        
    def calculate_parameter_errors(self, weights, target_params):
        """计算关键参数匹配误差（包含Duv）"""
        synthesized = self.synthesize_spectrum(weights)
        synthetic_params = self.calculator.calculate_all_parameters(
            self.wavelengths, synthesized)
        
        cct_error = 0
        mel_der_error = 0
        duv_error = 0
        
        # CCT相对误差
        if 'CCT' in target_params and target_params['CCT'] > 0:
            cct_error = abs(target_params['CCT'] - synthetic_params['CCT']) / target_params['CCT']
        
        # mel-DER相对误差
        if 'mel-DER' in target_params and target_params['mel-DER'] > 0:
            mel_der_error = abs(target_params['mel-DER'] - synthetic_params['mel-DER']) / target_params['mel-DER']
        
        # Duv相对误差
        if 'Duv' in target_params:
            duv_error = abs(target_params['Duv'] - synthetic_params['Duv']) / abs(target_params['Duv'])
        
        return cct_error + mel_der_error + duv_error
        
    def optimize_single_timepoint(self, target_spectrum):
        """对单个时间点优化权重"""
        def objective(weights):
            # 光谱匹配误差
            spectrum_error = self.calculate_spectrum_error(weights, target_spectrum)
            # 参数匹配误差（包含Duv）
            param_error = self.calculate_parameter_errors(weights, target_params)
            # 组合误差
            return alpha * spectrum_error + beta * param_error
            
        # 约束：权重和为1，非负
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(5)]
        
        result = minimize(objective, x0=[0.2]*5, 
                         constraints=constraints, bounds=bounds,
                         method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-9})
        return result.x
        
    def generate_control_sequence(self):
        """生成全天控制序列"""
        control_sequence = []
        for time_point, spectrum in self.solar_data.items():
            weights = self.optimize_single_timepoint(spectrum)
            control_sequence.append((time_point, weights))
        return control_sequence
```

### 6. 评估指标与验证方法

#### 6.1 光谱相似性评估
- **均方根误差（RMSE）：**
  $$RMSE = \sqrt{\frac{1}{n}\sum_{\lambda}[S_{\text{sun}}(\lambda) - S_{\text{合成}}(\lambda)]^2}$$

- **皮尔逊相关系数：**
  $$r = \frac{\sum(S_{\text{sun}} - \bar{S}_{\text{sun}})(S_{\text{合成}} - \bar{S}_{\text{合成}})}{\sqrt{\sum(S_{\text{sun}} - \bar{S}_{\text{sun}})^2\sum(S_{\text{合成}} - \bar{S}_{\text{合成}})^2}}$$

#### 6.2 关键参数匹配度
- CCT相对误差：$\frac{|CCT_{\text{target}} - CCT_{\text{synthetic}}|}{CCT_{\text{target}}} \times 100\%$
- Duv相对误差：$\frac{|Duv_{\text{target}} - Duv_{\text{synthetic}}|}{|Duv_{\text{target}}|} \times 100\%$
- mel-DER相对误差：$\frac{|mel\text{-}DER_{\text{target}} - mel\text{-}DER_{\text{synthetic}}|}{mel\text{-}DER_{\text{target}}} \times 100\%$

**Duv匹配精度说明：**
- Duv值通常在-0.02到+0.02之间
- 高质量照明要求Duv相对误差<10%
- 使用相对误差形式与其他参数保持一致，便于统一权重调整

### 7. 三个代表性时间点案例分析

选择三个具有显著差异的时间点：

#### 7.1 早晨（8:30）
- **特征**：低色温，暖光，mel-DER相对较低
- **预期策略**：主要使用暖白光WW，适量红光补充

#### 7.2 正午（12:30）  
- **特征**：高色温，冷光，mel-DER较高
- **预期策略**：主要使用冷白光CW，适量蓝光增强

#### 7.3 傍晚（18:30）
- **特征**：中等偏低色温，向暖光过渡
- **预期策略**：WW和CW混合，减少蓝光成分

### 8. 实现步骤总结

1. **数据预处理**：读取太阳光谱时间序列和LED通道数据
2. **目标参数计算**：为每个时间点计算CCT、Duv、mel-DER等关键参数
3. **单点优化**：使用约束优化算法求解每个时间点的最佳权重组合
4. **序列生成**：生成全天控制权重序列
5. **插值平滑**：在离散时间点间实现平滑过渡
6. **效果验证**：比较合成光谱与目标光谱的相似性
7. **可视化分析**：绘制代表性时间点的对比图
