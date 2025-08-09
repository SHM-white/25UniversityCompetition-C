# 问题四：基于睡眠实验数据的"优化光照"效果统计分析

## 1. 问题概述

### 1.1 研究背景
根据前序问题的理论设计，我们已经：
- **问题1**：建立了SPD到5个核心参数的标准化计算模型（CCT、Duv、Rf、Rg、mel-DER）
- **问题2**：设计了针对"夜间助眠模式"的最优光谱（低CCT=3000±500K，低mel-DER，Rf≥80）
- **问题3**：实现了全天候太阳光模拟的LED控制策略

现在需要通过临床睡眠实验数据验证理论设计的实际效果。

### 1.2 实验设计
- **实验类型**：交叉实验（Crossover Design）
- **被试数量**：11位健康被试
- **实验条件**：每位被试分别体验3种睡前光照环境
  - **环境A**：问题二设计的"夜间助眠模式"光谱（优化光照）
  - **环境B**：普通市售LED灯光（对照组1）
  - **环境C**：严格黑暗环境（对照组2）
- **光照时间**：睡前2小时
- **数据采集**：便携式睡眠监测仪，每30秒记录一次
- **总记录数**：11×3=33条有效睡眠记录

### 1.3 睡眠阶段编码标准（AASM）
- **4** - 清醒（Wake）
- **5** - REM睡眠（Rapid Eye Movement）
- **2** - N2期睡眠（Stage N2）+ N1期睡眠（Stage N1），即浅睡眠
- **3** - N3期睡眠（Stage N3），即深睡眠/慢波睡眠

## 2. 利用前三问建立的计算框架

### 2.1 基于问题1的参数计算能力
利用问题1建立的`SPDCalculator`类，我们已具备：

```python
# 从问题1继承的核心计算能力
class SPDCalculator:
    def calculate_cct_duv(self, wavelengths, spd_values)  # 色温和Duv计算
    def calculate_color_rendering(self, wavelengths, spd_values)  # Rf和Rg计算
    def calculate_mel_der(self, wavelengths, spd_values)  # mel-DER计算
```

这为验证"夜间助眠模式"的光谱特性提供了基础。

### 2.2 基于问题2的优化光照设计
问题2通过遗传算法优化得到的"夜间助眠模式"：
- **目标函数**：$\min(\text{mel-DER})$，最小化生理节律干扰
- **约束条件**：CCT=3000±500K，Rf≥80
- **权重组合**：$\mathbf{w}^* = [w_{\text{blue}}, w_{\text{green}}, w_{\text{red}}, w_{\text{warm}}, w_{\text{cool}}]$

### 2.3 基于问题3的光谱合成技术
问题3建立的多通道LED光谱合成公式：
$$S_{\text{合成}}(\lambda) = \sum_{i=1}^{5} w_i \times S_i(\lambda)$$

这为理解和验证优化光照的光谱特性提供了技术基础。

## 3. 睡眠质量评估指标计算模型

基于AASM标准和睡眠医学常用指标，建立以下计算模型：

### 3.1 总睡眠时间（TST - Total Sleep Time）
$$TST = \sum_{t} I_{sleep}(t) \times \Delta t$$

其中：
- $I_{sleep}(t) = \begin{cases} 1 & \text{if stage}(t) \in \{2,3,5\} \\ 0 & \text{if stage}(t) = 4 \end{cases}$
- $\Delta t = 0.5\text{分钟}$（每30秒记录一次）

```python
def calculate_tst(sleep_stages):
    """计算总睡眠时间（分钟）"""
    sleep_epochs = sum(1 for stage in sleep_stages if stage in [2, 3, 5])
    return sleep_epochs * 0.5  # 每个epoch = 30秒 = 0.5分钟
```

### 3.2 睡眠效率（SE - Sleep Efficiency）
$$SE = \frac{TST}{TIB} \times 100\%$$

其中$TIB$（Total time In Bed）为总卧床时间：
$$TIB = \text{len}(\text{sleep\_data}) \times \Delta t$$

```python
def calculate_sleep_efficiency(sleep_stages):
    """计算睡眠效率（百分比）"""
    tst = calculate_tst(sleep_stages)
    tib = len(sleep_stages) * 0.5
    return (tst / tib) * 100
```

### 3.3 入睡潜伏期（SOL - Sleep Onset Latency）
$$SOL = \text{first\_sleep\_epoch} \times \Delta t$$

```python
def calculate_sol(sleep_stages):
    """计算入睡潜伏期（分钟）"""
    for i, stage in enumerate(sleep_stages):
        if stage in [2, 3, 5]:  # 首次进入任何睡眠阶段
            return i * 0.5
    return len(sleep_stages) * 0.5  # 如果整夜未入睡
```

### 3.4 深睡眠比例（N3%）
$$N3\% = \frac{\sum_{t} I_{N3}(t) \times \Delta t}{TST} \times 100\%$$

其中：$I_{N3}(t) = \begin{cases} 1 & \text{if stage}(t) = 3 \\ 0 & \text{otherwise} \end{cases}$

```python
def calculate_n3_percentage(sleep_stages):
    """计算深睡眠比例（百分比）"""
    n3_time = sum(0.5 for stage in sleep_stages if stage == 3)
    tst = calculate_tst(sleep_stages)
    return (n3_time / tst) * 100 if tst > 0 else 0
```

### 3.5 REM睡眠比例（REM%）
$$REM\% = \frac{\sum_{t} I_{REM}(t) \times \Delta t}{TST} \times 100\%$$

```python
def calculate_rem_percentage(sleep_stages):
    """计算REM睡眠比例（百分比）"""
    rem_time = sum(0.5 for stage in sleep_stages if stage == 5)
    tst = calculate_tst(sleep_stages)
    return (rem_time / tst) * 100 if tst > 0 else 0
```

### 3.6 夜间醒来次数（Number of Awakenings）
$$\text{Awakenings} = \sum_{t=1}^{n-1} I_{transition}(t)$$

其中：$I_{transition}(t) = \begin{cases} 1 & \text{if stage}(t) \neq 4 \text{ and stage}(t+1) = 4 \text{ and } t > \text{SOL} \\ 0 & \text{otherwise} \end{cases}$

```python
def calculate_awakenings(sleep_stages):
    """计算夜间醒来次数"""
    if len(sleep_stages) < 2:
        return 0
    
    # 找到入睡时间点
    sol_epoch = None
    for i, stage in enumerate(sleep_stages):
        if stage in [2, 3, 5]:
            sol_epoch = i
            break
    
    if sol_epoch is None:
        return 0
    
    awakenings = 0
    for i in range(sol_epoch, len(sleep_stages) - 1):
        if sleep_stages[i] in [2, 3, 5] and sleep_stages[i + 1] == 4:
            awakenings += 1
    
    return awakenings
```

## 4. 统计分析策略

### 4.1 数据结构组织
```python
# 数据组织结构
sleep_data = {
    'subject_id': [1, 1, 1, 2, 2, 2, ...],  # 被试编号
    'condition': ['A', 'B', 'C', 'A', 'B', 'C', ...],  # 光照条件
    'tst': [...],           # 总睡眠时间
    'se': [...],            # 睡眠效率
    'sol': [...],           # 入睡潜伏期
    'n3_percent': [...],    # 深睡眠比例
    'rem_percent': [...],   # REM睡眠比例
    'awakenings': [...]     # 夜间醒来次数
}
```

### 4.2 统计假设检验前的前提条件验证

#### 4.2.1 正态性检验（Normality Tests）
使用Shapiro-Wilk检验验证各组数据的正态性：

$$H_0: \text{数据来自正态分布} \quad \text{vs} \quad H_1: \text{数据不来自正态分布}$$

```python
def normality_tests(self) -> Dict:
    """正态性检验"""
    normality_results = {}
    
    for metric in self.metrics:
        normality_results[metric] = {}
        for condition in ['A', 'B', 'C']:
            data = self.data[self.data['condition'] == condition][metric]
            stat, p_value = stats.shapiro(data)
            normality_results[metric][condition] = p_value
    
    return normality_results
```

#### 4.2.2 方差齐性检验（Homogeneity of Variance Tests）
检验各组间方差是否相等，这是进行ANOVA的重要前提：

$$H_0: \sigma_A^2 = \sigma_B^2 = \sigma_C^2 \quad \text{vs} \quad H_1: \text{方差不全相等}$$

**Levene检验**（推荐使用，对非正态分布较稳健）：
```python
def variance_homogeneity_tests(self) -> Dict:
    """方差齐性检验"""
    variance_results = {}
    
    for metric in self.metrics:
        # 准备各组数据
        groups = [self.data[self.data['condition'] == cond][metric] 
                 for cond in ['A', 'B', 'C']]
        
        # Levene检验（基于中位数）
        levene_stat, levene_p = stats.levene(*groups, center='median')
        
        # Bartlett检验（假设正态分布）
        bartlett_stat, bartlett_p = stats.bartlett(*groups)
        
        variance_results[metric] = {
            'levene': {
                'statistic': levene_stat,
                'p_value': levene_p,
                'homogeneous': levene_p > 0.05
            },
            'bartlett': {
                'statistic': bartlett_stat,
                'p_value': bartlett_p,
                'homogeneous': bartlett_p > 0.05
            }
        }
    
    return variance_results
```

### 4.3 统计检验方法选择决策树

基于前提条件检验结果，选择合适的统计方法：

```
数据正态性 + 方差齐性
├── 满足 → 重复测量方差分析（Repeated Measures ANOVA）
├── 正态但方差不齐 → Welch ANOVA 或 非参数检验
└── 不满足正态性 → Friedman非参数检验
```

#### 4.3.1 重复测量方差分析（Repeated Measures ANOVA）
适用于交叉实验设计，检验三种光照条件的主效应：

$$H_0: \mu_A = \mu_B = \mu_C \quad \text{vs} \quad H_1: \text{至少一对均值不等}$$

**前提条件**：
- ✓ 正态性：各组数据符合正态分布
- ✓ 方差齐性：各组方差相等
- ✓ 球形性：协方差矩阵满足球形性假设

```python
def repeated_measures_anova(self, metric: str) -> Dict:
    """重复测量方差分析"""
    result = pg.rm_anova(
        data=self.data,
        dv=metric,
        within='condition',
        subject='subject_id',
        detailed=True
    )
    return result
```

#### 4.3.2 Friedman检验（非参数替代）
当数据不满足正态性或方差齐性假设时：

$$H_0: \text{三种条件的中位数相等} \quad \text{vs} \quad H_1: \text{中位数不全相等}$$

```python
def friedman_test(self, metric: str) -> Tuple[float, float]:
    """Friedman非参数检验"""
    condition_a = self.data[self.data['condition'] == 'A'][metric].values
    condition_b = self.data[self.data['condition'] == 'B'][metric].values
    condition_c = self.data[self.data['condition'] == 'C'][metric].values
    
    statistic, p_value = stats.friedmanchisquare(
        condition_a, condition_b, condition_c
    )
    return statistic, p_value
```

#### 4.3.3 事后比较（Post-hoc Analysis）
显著性检验后的两两比较：

```python
def post_hoc_analysis(self, metric: str) -> pd.DataFrame:
    """事后比较分析"""
    post_hoc = pg.pairwise_tests(
        data=self.data,
        dv=metric,
        within='condition',
        subject='subject_id',
        padjust='bonf'  # Bonferroni校正
    )
    return post_hoc
```

### 4.4 效应量计算
$$\eta^2 = \frac{SS_{condition}}{SS_{total}}$$

$$\text{Cohen's d} = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}}$$

**效应量解释标准**：
- Cohen's d < 0.2：很小效应
- 0.2 ≤ Cohen's d < 0.5：小效应  
- 0.5 ≤ Cohen's d < 0.8：中等效应
- Cohen's d ≥ 0.8：大效应

```python
def effect_size_calculation(self, metric: str) -> Dict[str, float]:
    """计算效应量"""
    effect_sizes = {}
    conditions = ['A', 'B', 'C']
    
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            cond1, cond2 = conditions[i], conditions[j]
            data1 = self.data[self.data['condition'] == cond1][metric]
            data2 = self.data[self.data['condition'] == cond2][metric]
            
            # Cohen's d
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                (len(data2) - 1) * data2.var()) / 
                               (len(data1) + len(data2) - 2))
            
            if pooled_std > 0:
                cohens_d = (data1.mean() - data2.mean()) / pooled_std
                effect_sizes[f'{cond1}_vs_{cond2}'] = cohens_d
    
    return effect_sizes
```

## 5. 综合分析流程实现

### 5.1 数据预处理
1. **读取睡眠数据**：从附录4.xlsx提取11×3=33条记录
2. **数据清洗**：处理缺失值和异常值
3. **指标计算**：计算6个睡眠质量指标
4. **数据验证**：确保数据完整性和一致性

### 5.2 前提条件检验
```python
def comprehensive_analysis(self) -> Dict:
    """综合统计分析"""
    # 1. 描述性统计
    descriptive = self.descriptive_statistics()
    
    # 2. 正态性检验
    normality = self.normality_tests()
    
    # 3. 方差齐性检验
    variance_homogeneity = self.variance_homogeneity_tests()
    
    analysis_results = {
        'descriptive': descriptive,
        'normality': normality,
        'variance_homogeneity': variance_homogeneity,
        'anova_results': {},
        'friedman_results': {},
        'post_hoc_results': {},
        'effect_sizes': {}
    }
```

### 5.3 统计方法决策流程
```python
# 对每个指标进行分析
for metric in self.metrics:
    # 判断是否使用参数或非参数检验
    metric_normality = normality[metric]
    metric_variance = variance_homogeneity[metric]
    all_normal = all(p > 0.05 for p in metric_normality.values())
    variance_homogeneous = metric_variance['levene']['homogeneous']
    
    if all_normal and variance_homogeneous:
        print("✓ 满足ANOVA所有假设，使用重复测量方差分析...")
        anova_result = self.repeated_measures_anova(metric)
        
        # 如果显著，进行事后比较
        if anova_result['p-unc'].iloc[0] < 0.05:
            post_hoc = self.post_hoc_analysis(metric)
            
    elif all_normal and not variance_homogeneous:
        print("⚠ 数据正态但方差不齐，使用Friedman非参数检验...")
        friedman_stat, friedman_p = self.friedman_test(metric)
        
    else:
        print("⚠ 数据不满足正态性假设，使用Friedman非参数检验...")
        friedman_stat, friedman_p = self.friedman_test(metric)
```

### 5.4 描述性统计
```python
def descriptive_statistics(self) -> pd.DataFrame:
    """计算描述性统计"""
    summary = self.data.groupby('condition')[self.metrics].agg([
        'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    # 中文标签映射
    condition_labels = {
        'A': '优化光照（夜间助眠模式）',
        'B': '普通LED光照', 
        'C': '黑暗环境'
    }
    
    return summary
```

### 5.5 假设检验框架
**原假设**：三种光照条件对睡眠质量指标无显著影响
**备择假设**：至少一种光照条件对睡眠质量有显著影响

**显著性水平**：α = 0.05
**多重比较校正**：Bonferroni校正

### 5.6 结果解释框架
基于统计检验结果，预期能够回答：

1. **前提条件验证**：
   - 各指标是否满足正态性假设？
   - 方差齐性检验结果如何？
   - 应选择参数还是非参数检验？

2. **主效应检验**：
   - 三种光照条件是否存在显著差异？
   - F统计量或χ²统计量及其p值
   - 效应量大小及其临床意义

3. **两两比较**（如主效应显著）：
   - 优化光照vs普通光照：环境A是否显著优于环境B？
   - 优化光照vs黑暗环境：环境A是否显著优于环境C？
   - 普通光照vs黑暗环境：环境B与环境C的差异？

4. **效应量评估**：
   - Cohen's d值的大小和解释
   - 统计显著性是否具有临床意义？
   - 效应量是否达到实际应用价值？

## 6. 理论验证意义

### 6.1 问题2设计验证
通过统计分析验证问题2中"夜间助眠模式"的实际效果：
- **低mel-DER设计**是否真正改善睡眠？
- **约束优化结果**是否在实际应用中有效？

### 6.2 问题3模拟效果验证
验证问题3中太阳光模拟技术的节律调节效果：
- **光谱合成技术**是否产生预期的生理效应？
- **多目标优化策略**是否在实际中可行？

### 6.3 问题1计算模型验证
验证问题1建立的参数计算模型的预测能力：
- **mel-DER指标**是否能有效预测睡眠质量？
- **CCT和显色性**的平衡是否合理？

## 7. 预期结论框架

基于统计分析结果，预期得出以下形式的结论：

**如果统计检验显著**：
> "基于重复测量方差分析，三种光照条件对[具体指标]存在显著影响（F(2,20)=X.XX, p<0.05）。事后比较显示，优化光照（环境A）相比普通光照（环境B）在[具体指标]上有显著改善（p<0.05, Cohen's d=X.XX），表明问题二设计的'夜间助眠模式'对改善睡眠质量产生了有益的效果。"

**如果统计检验不显著**：
> "统计分析未发现三种光照条件对睡眠质量指标的显著影响，可能需要增加样本量或调整光照参数设计以获得更明确的效果。"

这一分析将为光照对人体生理节律影响的理论研究提供重要的实证支撑，并为智能照明系统的实际应用提供科学依据。
