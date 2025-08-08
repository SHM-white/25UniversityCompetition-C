"""
问题四：基于睡眠实验数据的"优化光照"效果统计分析

实现睡眠质量评估指标计算和统计分析，验证问题二设计的"夜间助眠模式"的实际效果。

基于前三问的计算框架：
- 问题1：SPD参数计算模型
- 问题2：夜间助眠模式设计
- 问题3：光谱合成技术
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pingouin as pg

# 设置中文显示和警告过滤
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans', '微软雅黑']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


@dataclass
class SleepMetrics:
    """睡眠质量指标数据类"""
    subject_id: int
    condition: str  # 'A', 'B', 'C'
    tst: float      # 总睡眠时间（分钟）
    se: float       # 睡眠效率（%）
    sol: float      # 入睡潜伏期（分钟）
    n3_percent: float   # 深睡眠比例（%）
    rem_percent: float  # REM睡眠比例（%）
    awakenings: int     # 夜间醒来次数

class SleepDataProcessor:
    """睡眠数据处理和分析类"""
    
    def __init__(self, excel_path: str = "C/附录4.xlsx"):
        """
        初始化睡眠数据处理器
        
        Parameters:
        excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.raw_data = None
        self.sleep_metrics = []
        self.conditions_map = {
            0: 'A',  # Night 1: 优化光照（夜间助眠模式）
            1: 'B',  # Night 2: 普通LED光照
            2: 'C'   # Night 3: 黑暗环境
        }
        
    def load_data(self) -> pd.DataFrame:
        """加载睡眠数据"""
        try:
            self.raw_data = pd.read_excel(self.excel_path, sheet_name='Problem 4')
            print(f"成功加载睡眠数据，形状: {self.raw_data.shape}")
            print(f"数据列数: {self.raw_data.shape[1]}，预期被试数: {self.raw_data.shape[1]//3}")
            return self.raw_data
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def extract_subject_data(self, subject_idx: int) -> List[List[int]]:
        """
        提取单个被试的三夜睡眠数据
        
        Parameters:
        subject_idx: 被试索引（0-10）
        
        Returns:
        三夜睡眠数据列表
        """
        col_start = subject_idx * 3
        subject_data = []
        
        for night in range(3):
            col_idx = col_start + night
            if col_idx < self.raw_data.shape[1]:
                # 跳过标题行，提取睡眠阶段数据
                night_data = self.raw_data.iloc[1:, col_idx].dropna().tolist()
                # 转换为整数，过滤无效数据
                night_data = [int(x) for x in night_data if pd.notna(x) and x in [2, 3, 4, 5]]
                subject_data.append(night_data)
            else:
                subject_data.append([])
        
        return subject_data
    
    def calculate_tst(self, sleep_stages: List[int]) -> float:
        """
        计算总睡眠时间（分钟）
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        总睡眠时间（分钟）
        """
        sleep_epochs = sum(1 for stage in sleep_stages if stage in [2, 3, 5])
        return sleep_epochs * 0.5  # 每个epoch = 30秒 = 0.5分钟
    
    def calculate_sleep_efficiency(self, sleep_stages: List[int]) -> float:
        """
        计算睡眠效率（百分比）
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        睡眠效率（%）
        """
        tst = self.calculate_tst(sleep_stages)
        tib = len(sleep_stages) * 0.5  # 总卧床时间
        return (tst / tib) * 100 if tib > 0 else 0
    
    def calculate_sol(self, sleep_stages: List[int]) -> float:
        """
        计算入睡潜伏期（分钟）
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        入睡潜伏期（分钟）
        """
        for i, stage in enumerate(sleep_stages):
            if stage in [2, 3, 5]:  # 首次进入任何睡眠阶段
                return i * 0.5
        return len(sleep_stages) * 0.5  # 如果整夜未入睡
    
    def calculate_n3_percentage(self, sleep_stages: List[int]) -> float:
        """
        计算深睡眠比例（百分比）
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        深睡眠比例（%）
        """
        n3_time = sum(0.5 for stage in sleep_stages if stage == 3)
        tst = self.calculate_tst(sleep_stages)
        return (n3_time / tst) * 100 if tst > 0 else 0
    
    def calculate_rem_percentage(self, sleep_stages: List[int]) -> float:
        """
        计算REM睡眠比例（百分比）
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        REM睡眠比例（%）
        """
        rem_time = sum(0.5 for stage in sleep_stages if stage == 5)
        tst = self.calculate_tst(sleep_stages)
        return (rem_time / tst) * 100 if tst > 0 else 0
    
    def calculate_awakenings(self, sleep_stages: List[int]) -> int:
        """
        计算夜间醒来次数
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        
        Returns:
        醒来次数
        """
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
    
    def calculate_sleep_metrics(self, sleep_stages: List[int], subject_id: int, condition: str) -> SleepMetrics:
        """
        计算完整的睡眠质量指标
        
        Parameters:
        sleep_stages: 睡眠阶段序列
        subject_id: 被试ID
        condition: 实验条件
        
        Returns:
        睡眠质量指标对象
        """
        return SleepMetrics(
            subject_id=subject_id,
            condition=condition,
            tst=self.calculate_tst(sleep_stages),
            se=self.calculate_sleep_efficiency(sleep_stages),
            sol=self.calculate_sol(sleep_stages),
            n3_percent=self.calculate_n3_percentage(sleep_stages),
            rem_percent=self.calculate_rem_percentage(sleep_stages),
            awakenings=self.calculate_awakenings(sleep_stages)
        )
    
    def process_all_data(self) -> List[SleepMetrics]:
        """处理所有被试的睡眠数据"""
        if self.raw_data is None:
            self.load_data()
        
        self.sleep_metrics = []
        n_subjects = min(11, self.raw_data.shape[1] // 3)  # 最多11个被试
        
        print(f"开始处理{n_subjects}个被试的睡眠数据...")
        
        for subject_idx in range(n_subjects):
            subject_id = subject_idx + 1
            subject_data = self.extract_subject_data(subject_idx)
            
            for night_idx, night_data in enumerate(subject_data):
                if len(night_data) > 0:  # 确保有有效数据
                    condition = self.conditions_map[night_idx]
                    metrics = self.calculate_sleep_metrics(night_data, subject_id, condition)
                    self.sleep_metrics.append(metrics)
                    
                    print(f"被试{subject_id}, 条件{condition}: TST={metrics.tst:.1f}min, "
                          f"SE={metrics.se:.1f}%, SOL={metrics.sol:.1f}min")
        
        print(f"\n总共处理了{len(self.sleep_metrics)}条睡眠记录")
        return self.sleep_metrics
    
    def get_dataframe(self) -> pd.DataFrame:
        """将睡眠指标转换为DataFrame"""
        if not self.sleep_metrics:
            self.process_all_data()
        
        data_dict = {
            'subject_id': [m.subject_id for m in self.sleep_metrics],
            'condition': [m.condition for m in self.sleep_metrics],
            'tst': [m.tst for m in self.sleep_metrics],
            'se': [m.se for m in self.sleep_metrics],
            'sol': [m.sol for m in self.sleep_metrics],
            'n3_percent': [m.n3_percent for m in self.sleep_metrics],
            'rem_percent': [m.rem_percent for m in self.sleep_metrics],
            'awakenings': [m.awakenings for m in self.sleep_metrics]
        }
        
        return pd.DataFrame(data_dict)

class SleepStatisticalAnalyzer:
    """睡眠数据统计分析类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化统计分析器
        
        Parameters:
        data: 睡眠指标DataFrame
        """
        self.data = data
        self.metrics = ['tst', 'se', 'sol', 'n3_percent', 'rem_percent', 'awakenings']
        self.condition_labels = {
            'A': '优化光照（夜间助眠模式）',
            'B': '普通LED光照',
            'C': '黑暗环境'
        }
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """计算描述性统计"""
        print("=== 描述性统计分析 ===")
        
        summary = self.data.groupby('condition')[self.metrics].agg(['mean', 'std', 'median']).round(2)
        
        # 打印格式化的结果
        for condition in ['A', 'B', 'C']:
            print(f"\n{condition}组 ({self.condition_labels[condition]}):")
            condition_data = self.data[self.data['condition'] == condition]
            print(f"样本量: {len(condition_data)}")
            
            for metric in self.metrics:
                mean_val = condition_data[metric].mean()
                std_val = condition_data[metric].std()
                median_val = condition_data[metric].median()
                print(f"  {metric}: 均值={mean_val:.2f}±{std_val:.2f}, 中位数={median_val:.2f}")
        
        return summary
    
    def normality_tests(self) -> Dict[str, Dict[str, float]]:
        """正态性检验"""
        print("\n=== 正态性检验（Shapiro-Wilk Test）===")
        
        normality_results = {}
        
        for metric in self.metrics:
            normality_results[metric] = {}
            print(f"\n{metric}:")
            
            for condition in ['A', 'B', 'C']:
                condition_data = self.data[self.data['condition'] == condition][metric]
                statistic, p_value = stats.shapiro(condition_data)
                normality_results[metric][condition] = p_value
                
                normal_status = "正态分布" if p_value > 0.05 else "非正态分布"
                print(f"  {condition}组: W={statistic:.4f}, p={p_value:.4f} ({normal_status})")
        
        return normality_results
    
    def repeated_measures_anova(self, metric: str) -> Dict:
        """重复测量方差分析"""
        try:
            # 使用pingouin进行重复测量方差分析
            result = pg.rm_anova(
                data=self.data,
                dv=metric,
                within='condition',
                subject='subject_id',
                detailed=True
            )
            return result
        except Exception as e:
            print(f"重复测量方差分析出错: {e}")
            return None
    
    def friedman_test(self, metric: str) -> Tuple[float, float]:
        """Friedman非参数检验"""
        # 准备数据
        condition_a = self.data[self.data['condition'] == 'A'][metric].values
        condition_b = self.data[self.data['condition'] == 'B'][metric].values
        condition_c = self.data[self.data['condition'] == 'C'][metric].values
        
        # 确保数据长度一致
        min_length = min(len(condition_a), len(condition_b), len(condition_c))
        condition_a = condition_a[:min_length]
        condition_b = condition_b[:min_length]
        condition_c = condition_c[:min_length]
        
        statistic, p_value = stats.friedmanchisquare(condition_a, condition_b, condition_c)
        return statistic, p_value
    
    def post_hoc_analysis(self, metric: str) -> pd.DataFrame:
        """事后比较分析"""
        try:
            # 使用pingouin进行事后比较
            post_hoc = pg.pairwise_tests(
                data=self.data,
                dv=metric,
                within='condition',
                subject='subject_id',
                padjust='bonf'  # Bonferroni校正
            )
            return post_hoc
        except Exception as e:
            print(f"事后比较分析出错: {e}")
            return None
    
    def effect_size_calculation(self, metric: str) -> Dict[str, float]:
        """计算效应量"""
        effect_sizes = {}
        
        # Cohen's d for pairwise comparisons
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
    
    def comprehensive_analysis(self) -> Dict:
        """综合统计分析"""
        print("=" * 60)
        print("睡眠质量指标综合统计分析")
        print("=" * 60)
        
        # 描述性统计
        descriptive = self.descriptive_statistics()
        
        # 正态性检验
        normality = self.normality_tests()
        
        analysis_results = {
            'descriptive': descriptive,
            'normality': normality,
            'anova_results': {},
            'friedman_results': {},
            'post_hoc_results': {},
            'effect_sizes': {}
        }
        
        # 对每个指标进行分析
        for metric in self.metrics:
            print(f"\n{'='*40}")
            print(f"分析指标: {metric}")
            print(f"{'='*40}")
            
            # 判断是否使用参数或非参数检验
            metric_normality = normality[metric]
            all_normal = all(p > 0.05 for p in metric_normality.values())
            
            if all_normal:
                print("数据满足正态性假设，使用重复测量方差分析...")
                anova_result = self.repeated_measures_anova(metric)
                if anova_result is not None:
                    analysis_results['anova_results'][metric] = anova_result
                    print("重复测量方差分析结果:")
                    print(anova_result)
                    
                    # 如果显著，进行事后比较
                    if len(anova_result) > 0 and anova_result['p-unc'].iloc[0] < 0.05:
                        print("\n发现显著差异，进行事后比较...")
                        post_hoc = self.post_hoc_analysis(metric)
                        if post_hoc is not None:
                            analysis_results['post_hoc_results'][metric] = post_hoc
                            print("事后比较结果:")
                            print(post_hoc)
            else:
                print("数据不满足正态性假设，使用Friedman非参数检验...")
                friedman_stat, friedman_p = self.friedman_test(metric)
                analysis_results['friedman_results'][metric] = {
                    'statistic': friedman_stat,
                    'p_value': friedman_p
                }
                print(f"Friedman检验结果: χ²={friedman_stat:.4f}, p={friedman_p:.4f}")
            
            # 计算效应量
            effect_sizes = self.effect_size_calculation(metric)
            analysis_results['effect_sizes'][metric] = effect_sizes
            print(f"\n效应量（Cohen's d）:")
            for comparison, d in effect_sizes.items():
                effect_magnitude = self._interpret_cohens_d(d)
                print(f"  {comparison}: d={d:.3f} ({effect_magnitude})")
        
        return analysis_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应量大小"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "很小效应"
        elif abs_d < 0.5:
            return "小效应"
        elif abs_d < 0.8:
            return "中等效应"
        else:
            return "大效应"

class SleepVisualization:
    """睡眠数据可视化类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化可视化器
        
        Parameters:
        data: 睡眠指标DataFrame
        """
        self.data = data
        self.metrics = ['tst', 'se', 'sol', 'n3_percent', 'rem_percent', 'awakenings']
        self.metric_labels = {
            'tst': '总睡眠时间 (分钟)',
            'se': '睡眠效率 (%)',
            'sol': '入睡潜伏期 (分钟)',
            'n3_percent': '深睡眠比例 (%)',
            'rem_percent': 'REM睡眠比例 (%)',
            'awakenings': '夜间醒来次数'
        }
        self.condition_labels = {
            'A': '优化光照',
            'B': '普通光照',
            'C': '黑暗环境'
        }
    
    def plot_boxplots(self, save_path: str = "Pictures/sleep_metrics_boxplots.svg"):
        """绘制箱线图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            
            # 准备数据
            plot_data = []
            labels = []
            for condition in ['A', 'B', 'C']:
                condition_data = self.data[self.data['condition'] == condition][metric]
                plot_data.append(condition_data)
                labels.append(self.condition_labels[condition])
            
            # 绘制箱线图
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(self.metric_labels[metric], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"箱线图已保存至: {save_path}")
    
    def plot_individual_trajectories(self, save_path: str = "Pictures/individual_trajectories.svg"):
        """绘制个体轨迹图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            
            # 为每个被试绘制轨迹
            for subject in self.data['subject_id'].unique():
                subject_data = self.data[self.data['subject_id'] == subject]
                
                if len(subject_data) == 3:  # 确保有三个条件的数据
                    conditions = ['A', 'B', 'C']
                    values = [subject_data[subject_data['condition'] == c][metric].iloc[0] 
                             for c in conditions]
                    
                    ax.plot(conditions, values, 'o-', alpha=0.6, linewidth=1, markersize=4)
            
            # 绘制均值轨迹
            mean_values = [self.data[self.data['condition'] == c][metric].mean() 
                          for c in ['A', 'B', 'C']]
            ax.plot(['A', 'B', 'C'], mean_values, 'ro-', linewidth=3, markersize=8, 
                   label='组均值')
            
            ax.set_title(self.metric_labels[metric], fontsize=12, fontweight='bold')
            ax.set_xlabel('光照条件')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"个体轨迹图已保存至: {save_path}")
    
    def plot_correlation_heatmap(self, save_path: str = "Pictures/sleep_metrics_correlation.svg"):
        """绘制睡眠指标相关性热图"""
        # 计算相关性矩阵
        corr_matrix = self.data[self.metrics].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})
        
        plt.title('睡眠质量指标相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"相关性热图已保存至: {save_path}")

def main():
    """主函数：执行完整的睡眠数据分析流程"""
    print("=" * 80)
    print("问题四：基于睡眠实验数据的'优化光照'效果统计分析")
    print("=" * 80)
    
    # 1. 数据处理
    print("\n步骤1: 数据加载和处理")
    processor = SleepDataProcessor()
    processor.load_data()
    metrics = processor.process_all_data()
    df = processor.get_dataframe()
    
    print(f"\n处理完成，数据概览:")
    print(f"总记录数: {len(df)}")
    print(f"被试数量: {df['subject_id'].nunique()}")
    print(f"实验条件: {sorted(df['condition'].unique())}")
    
    # 2. 统计分析
    print("\n步骤2: 统计分析")
    analyzer = SleepStatisticalAnalyzer(df)
    results = analyzer.comprehensive_analysis()
    
    # 3. 可视化
    print("\n步骤3: 数据可视化")
    visualizer = SleepVisualization(df)
    visualizer.plot_boxplots()
    visualizer.plot_individual_trajectories()
    visualizer.plot_correlation_heatmap()
    
    # 4. 结论总结
    print("\n" + "=" * 60)
    print("分析结论总结")
    print("=" * 60)
    
    significant_metrics = []
    
    # 检查显著性结果
    for metric in analyzer.metrics:
        is_significant = False
        
        # 检查ANOVA结果
        if metric in results['anova_results']:
            anova_result = results['anova_results'][metric]
            if len(anova_result) > 0 and anova_result['p-unc'].iloc[0] < 0.05:
                is_significant = True
        
        # 检查Friedman结果
        if metric in results['friedman_results']:
            friedman_result = results['friedman_results'][metric]
            if friedman_result['p_value'] < 0.05:
                is_significant = True
        
        if is_significant:
            significant_metrics.append(metric)
    
    if significant_metrics:
        print(f"发现显著差异的指标: {', '.join(significant_metrics)}")
        print("\n这表明'优化光照'设计对睡眠质量产生了可测量的影响。")
    else:
        print("未发现显著的组间差异。")
        print("可能需要更大的样本量或调整实验设计参数。")
    
    # 保存结果
    print(f"\n分析结果已保存至Pictures目录")
    
    return df, results

if __name__ == "__main__":
    # 执行主要分析流程
    sleep_data, analysis_results = main()
