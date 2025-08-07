"""
遗传算法优化器用于问题二：多通道LED光源优化
解决两种场景：
1. 日间照明模式：优化Rf，约束CCT在6000±500K，Rg在95-105
2. 夜间助眠模式：最小化mel-DER，约束CCT在3000±500K，Rf≥80
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import random
from spd_calculator import SPDCalculator, load_spd_from_excel
import warnings
warnings.filterwarnings('ignore')

class LEDOptimizer:
    def __init__(self, led_data_path="C/附录2_LED_SPD.xlsx"):
        """
        初始化LED优化器
        
        Parameters:
        led_data_path: LED SPD数据文件路径
        """
        self.calculator = SPDCalculator()
        self.load_led_data(led_data_path)
        
    def load_led_data(self, led_data_path):
        """加载5个LED通道的SPD数据"""
        try:
            # 读取Excel文件
            df = pd.read_excel(led_data_path, sheet_name=0, header=0)
            
            # 提取波长信息
            wavelength_col = df.iloc[:, 0]  # 第一列是波长
            self.wavelengths = []
            
            # 从波长列提取数值
            for i, wl_str in enumerate(wavelength_col):
                try:
                    if pd.isna(wl_str):
                        continue
                    wl_str = str(wl_str)
                    if '(' in wl_str:
                        wl = float(wl_str.split('(')[0])
                    else:
                        wl = float(wl_str)
                    self.wavelengths.append(wl)
                except:
                    continue
            
            self.wavelengths = np.array(self.wavelengths)
            
            # 提取5个LED通道的SPD数据
            self.led_channels = {}
            channel_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
            
            for ch_name in channel_names:
                if ch_name in df.columns:
                    spd_data = df[ch_name].values[:len(self.wavelengths)]
                    # 确保数据为数值类型
                    spd_data = pd.to_numeric(spd_data, errors='coerce')
                    spd_data = np.nan_to_num(spd_data, 0)  # 将NaN替换为0
                    self.led_channels[ch_name] = spd_data
                    print(f"加载{ch_name}通道：{len(spd_data)}个数据点，范围{spd_data.min():.3f}-{spd_data.max():.3f}")
            
            print(f"成功加载LED数据，波长范围: {self.wavelengths[0]:.0f}-{self.wavelengths[-1]:.0f}nm")
            
        except Exception as e:
            print(f"加载LED数据时出错: {e}")
            raise
    
    def synthesize_spd(self, weights):
        """
        合成光谱：各通道SPD的加权线性叠加
        
        Parameters:
        weights: 长度为5的权重数组 [Blue, Green, Red, Warm White, Cold White]
        
        Returns:
        tuple: (wavelengths, synthesized_spd)
        """
        channel_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
        
        # 初始化合成光谱
        synthesized_spd = np.zeros_like(self.wavelengths, dtype=float)
        
        # 加权叠加
        for i, ch_name in enumerate(channel_names):
            if i < len(weights) and ch_name in self.led_channels:
                synthesized_spd += weights[i] * self.led_channels[ch_name]
        
        return self.wavelengths, synthesized_spd
    
    def objective_function_daylight(self, weights):
        """
        日间照明模式的目标函数
        目标：最大化Rf
        约束：CCT在6000±500K，Rg在95-105，Rf>88
        
        Parameters:
        weights: 权重数组 [Blue, Green, Red, Warm White, Cold White]
        
        Returns:
        float: 目标函数值（越小越好）
        """
        try:
            # 权重归一化
            weights = np.array(weights)
            if np.sum(weights) == 0:
                return 1000  # 无效权重
            weights = weights / np.sum(weights)
            
            # 合成光谱
            wavelengths, spd = self.synthesize_spd(weights)
            
            if np.sum(spd) == 0:
                return 1000  # 无效光谱
            
            # 计算参数
            results = self.calculator.calculate_all_parameters(wavelengths, spd)
            
            cct = results['CCT']
            rf = results['Rf']
            rg = results['Rg']
            
            # 目标函数：最大化Rf（转为最小化问题）
            objective = -rf
            
            # 约束条件的惩罚项
            penalty = 0
            
            # CCT约束：6000±500K
            if cct < 5500 or cct > 6500:
                penalty += 100 * abs(cct - 6000) / 1000
            
            # Rg约束：95-105
            if rg < 95:
                penalty += 50 * (95 - rg)
            elif rg > 105:
                penalty += 50 * (rg - 105)
            
            # Rf约束：>88
            if rf < 88:
                penalty += 100 * (88 - rf)
            
            return objective + penalty
            
        except Exception as e:
            print(f"计算日间模式目标函数时出错: {e}")
            return 1000
    
    def objective_function_night(self, weights):
        """
        夜间助眠模式的目标函数
        目标：最小化mel-DER
        约束：CCT在3000±500K，Rf≥80
        
        Parameters:
        weights: 权重数组 [Blue, Green, Red, Warm White, Cold White]
        
        Returns:
        float: 目标函数值（越小越好）
        """
        try:
            # 权重归一化
            weights = np.array(weights)
            if np.sum(weights) == 0:
                return 1000  # 无效权重
            weights = weights / np.sum(weights)
            
            # 合成光谱
            wavelengths, spd = self.synthesize_spd(weights)
            
            if np.sum(spd) == 0:
                return 1000  # 无效光谱
            
            # 计算参数
            results = self.calculator.calculate_all_parameters(wavelengths, spd)
            
            cct = results['CCT']
            rf = results['Rf']
            mel_der = results['mel-DER']
            
            # 目标函数：最小化mel-DER
            objective = mel_der
            
            # 约束条件的惩罚项
            penalty = 0
            
            # CCT约束：3000±500K
            if cct < 2500 or cct > 3500:
                penalty += 100 * abs(cct - 3000) / 1000
            
            # Rf约束：≥80
            if rf < 80:
                penalty += 100 * (80 - rf)
            
            return objective + penalty
            
        except Exception as e:
            print(f"计算夜间模式目标函数时出错: {e}")
            return 1000

class GeneticAlgorithm:
    def __init__(self, optimizer, population_size=100, generations=200, mutation_rate=0.1, crossover_rate=0.7):
        """
        遗传算法类
        
        Parameters:
        optimizer: LEDOptimizer实例
        population_size: 种群大小
        generations: 进化代数
        mutation_rate: 变异率
        crossover_rate: 交叉率
        """
        self.optimizer = optimizer
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 随机生成5个通道的权重 [0, 1]
            individual = np.random.random(5)
            population.append(individual)
        return np.array(population)
    
    def evaluate_fitness(self, population, mode='daylight'):
        """评估种群适应度"""
        fitness = []
        for individual in population:
            if mode == 'daylight':
                obj_value = self.optimizer.objective_function_daylight(individual)
            elif mode == 'night':
                obj_value = self.optimizer.objective_function_night(individual)
            else:
                raise ValueError("模式必须是'daylight'或'night'")
            
            # 转换为适应度（目标函数值越小，适应度越高）
            fitness.append(1.0 / (1.0 + obj_value))
        
        return np.array(fitness)
    
    def selection(self, population, fitness):
        """轮盘赌选择"""
        selected = []
        total_fitness = np.sum(fitness)
        
        if total_fitness == 0:
            # 如果总适应度为0，随机选择
            return population[np.random.choice(len(population), size=len(population))]
        
        probabilities = fitness / total_fitness
        
        for _ in range(len(population)):
            r = np.random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(population[i].copy())
                    break
            else:
                selected.append(population[-1].copy())
        
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """单点交叉"""
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        """高斯变异"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
                mutated[i] = max(0, min(1, mutated[i]))  # 限制在[0,1]范围内
        return mutated
    
    def optimize(self, mode='daylight', verbose=True):
        """
        运行遗传算法优化
        
        Parameters:
        mode: 'daylight' 或 'night'
        verbose: 是否打印进度
        
        Returns:
        tuple: (best_weights, best_fitness, fitness_history)
        """
        print(f"\n开始{mode}模式的遗传算法优化...")
        print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
        
        # 初始化种群
        population = self.initialize_population()
        fitness_history = []
        
        for generation in range(self.generations):
            # 评估适应度
            fitness = self.evaluate_fitness(population, mode)
            
            # 记录最佳适应度
            best_fitness = np.max(fitness)
            fitness_history.append(best_fitness)
            
            if verbose and generation % 20 == 0:
                best_obj = 1.0 / best_fitness - 1.0
                print(f"第{generation}代: 最佳目标函数值 = {best_obj:.4f}")
            
            # 选择
            selected_population = self.selection(population, fitness)
            
            # 交叉和变异
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.population_size])
        
        # 返回最佳解
        final_fitness = self.evaluate_fitness(population, mode)
        best_idx = np.argmax(final_fitness)
        best_weights = population[best_idx]
        best_fitness = final_fitness[best_idx]
        
        return best_weights, best_fitness, fitness_history

def analyze_solution(optimizer, weights, mode_name):
    """分析优化解的性能"""
    print(f"\n=== {mode_name}模式优化结果分析 ===")
    
    # 权重归一化
    weights_norm = weights / np.sum(weights)
    print("通道权重:")
    channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
    for i, name in enumerate(channel_names):
        print(f"  {name}: {weights_norm[i]:.4f} ({weights_norm[i]*100:.1f}%)")
    
    # 合成光谱并计算参数
    wavelengths, spd = optimizer.synthesize_spd(weights_norm)
    results = optimizer.calculator.calculate_all_parameters(wavelengths, spd)
    
    # 打印结果
    optimizer.calculator.print_results(results)
    
    return results, wavelengths, spd

def plot_optimization_history(fitness_history_day, fitness_history_night):
    """绘制优化历史"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history_day)
    plt.title('日间照明模式优化历史')
    plt.xlabel('进化代数')
    plt.ylabel('适应度')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(fitness_history_night)
    plt.title('夜间助眠模式优化历史')
    plt.xlabel('进化代数')
    plt.ylabel('适应度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_synthesized_spectra(optimizer, weights_day, weights_night):
    """绘制合成光谱对比"""
    plt.figure(figsize=(12, 8))
    
    # 日间模式光谱
    weights_day_norm = weights_day / np.sum(weights_day)
    wavelengths_day, spd_day = optimizer.synthesize_spd(weights_day_norm)
    
    # 夜间模式光谱
    weights_night_norm = weights_night / np.sum(weights_night)
    wavelengths_night, spd_night = optimizer.synthesize_spd(weights_night_norm)
    
    # 上图：合成光谱对比
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths_day, spd_day, 'b-', linewidth=2, label='日间照明模式')
    plt.plot(wavelengths_night, spd_night, 'r-', linewidth=2, label='夜间助眠模式')
    plt.xlabel('波长 (nm)')
    plt.ylabel('相对光谱功率')
    plt.title('优化后的合成光谱对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下图：各通道贡献
    plt.subplot(2, 1, 2)
    channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
    colors = ['blue', 'green', 'red', 'orange', 'lightblue']
    
    x = np.arange(len(channel_names))
    width = 0.35
    
    plt.bar(x - width/2, weights_day_norm, width, label='日间模式', alpha=0.7, color=colors)
    plt.bar(x + width/2, weights_night_norm, width, label='夜间模式', alpha=0.7, color=colors)
    
    plt.xlabel('LED通道')
    plt.ylabel('权重')
    plt.title('各通道权重对比')
    plt.xticks(x, channel_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthesized_spectra_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 多通道LED光源遗传算法优化 ===")
    
    # 创建优化器
    optimizer = LEDOptimizer()
    
    # 创建遗传算法实例
    ga = GeneticAlgorithm(
        optimizer=optimizer,
        population_size=100,
        generations=200,
        mutation_rate=0.1,
        crossover_rate=0.7
    )
    
    # 优化日间照明模式
    print("\n" + "="*50)
    print("场景一：日间照明模式优化")
    print("目标：最大化Rf")
    print("约束：CCT在6000±500K，Rg在95-105，Rf>88")
    weights_day, fitness_day, history_day = ga.optimize(mode='daylight')
    results_day, wavelengths_day, spd_day = analyze_solution(optimizer, weights_day, "日间照明")
    
    # 优化夜间助眠模式
    print("\n" + "="*50)
    print("场景二：夜间助眠模式优化")
    print("目标：最小化mel-DER")
    print("约束：CCT在3000±500K，Rf≥80")
    weights_night, fitness_night, history_night = ga.optimize(mode='night')
    results_night, wavelengths_night, spd_night = analyze_solution(optimizer, weights_night, "夜间助眠")
    
    # 绘制结果
    plot_optimization_history(history_day, history_night)
    plot_synthesized_spectra(optimizer, weights_day, weights_night)
    
    # 总结对比
    print("\n" + "="*50)
    print("两种模式对比总结")
    print(f"日间模式 - CCT: {results_day['CCT']:.1f}K, Rf: {results_day['Rf']:.1f}, Rg: {results_day['Rg']:.1f}, mel-DER: {results_day['mel-DER']:.3f}")
    print(f"夜间模式 - CCT: {results_night['CCT']:.1f}K, Rf: {results_night['Rf']:.1f}, Rg: {results_night['Rg']:.1f}, mel-DER: {results_night['mel-DER']:.3f}")
    
    # 保存结果
    results_summary = {
        '模式': ['日间照明', '夜间助眠'],
        'CCT(K)': [results_day['CCT'], results_night['CCT']],
        'Duv': [results_day['Duv'], results_night['Duv']],
        'Rf': [results_day['Rf'], results_night['Rf']],
        'Rg': [results_day['Rg'], results_night['Rg']],
        'mel-DER': [results_day['mel-DER'], results_night['mel-DER']],
        '蓝光权重': [weights_day[0]/np.sum(weights_day), weights_night[0]/np.sum(weights_night)],
        '绿光权重': [weights_day[1]/np.sum(weights_day), weights_night[1]/np.sum(weights_night)],
        '红光权重': [weights_day[2]/np.sum(weights_day), weights_night[2]/np.sum(weights_night)],
        '暖白权重': [weights_day[3]/np.sum(weights_day), weights_night[3]/np.sum(weights_night)],
        '冷白权重': [weights_day[4]/np.sum(weights_day), weights_night[4]/np.sum(weights_night)]
    }
    
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv('optimization_results.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存到 optimization_results.csv")
    
    return optimizer, weights_day, weights_night, results_day, results_night

if __name__ == "__main__":
    optimizer, weights_day, weights_night, results_day, results_night = main()
