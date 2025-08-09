"""
问题三：全天候太阳光模拟LED控制策略
使用5通道LED系统模拟从早晨8:30到傍晚19:30的太阳光谱数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings
from tqdm import tqdm
import os
from datetime import datetime, timedelta
from spd_calculator import SPDCalculator
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', '微软雅黑', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SolarSpectrumMimicry:
    def __init__(self, led_data_path="C/附录2_LED_SPD.xlsx", solar_data_path="C/附录3_SUN_SPD.xlsx"):
        """
        初始化太阳光谱模拟器
        
        Parameters:
        led_data_path: LED SPD数据文件路径
        solar_data_path: 太阳光谱时间序列数据文件路径
        """
        self.calculator = SPDCalculator()
        self.load_led_data(led_data_path)
        self.load_solar_data(solar_data_path)
        self.optimization_results = {}
        
        # 确保输出文件夹存在
        if not os.path.exists('Pictures'):
            os.makedirs('Pictures')
        if not os.path.exists('DataFrames'):
            os.makedirs('DataFrames')
    
    def load_led_data(self, led_data_path):
        """加载5个LED通道的SPD数据"""
        print("Loading LED channel data...")
        try:
            df = pd.read_excel(led_data_path, sheet_name=0, header=0)
            
            # 提取波长信息（第一列）
            wavelength_col = df.iloc[:, 0]
            self.wavelengths = []
            
            # 从波长列提取数值
            for i, wl_str in enumerate(wavelength_col):
                if pd.isna(wl_str):
                    break
                try:
                    # 处理格式如 "380(mW/m2/nm)" 的情况
                    if isinstance(wl_str, str):
                        # 提取数值部分（括号之前）
                        wl_clean = wl_str.split('(')[0].strip()
                        wl = float(wl_clean)
                    elif isinstance(wl_str, (int, float)):
                        wl = float(wl_str)
                    else:
                        continue
                    
                    if 380 <= wl <= 780:  # 只保留可见光范围
                        self.wavelengths.append(wl)
                    else:
                        break
                except Exception as e:
                    print(f"Error parsing wavelength at row {i}: {wl_str}, error: {e}")
                    break
            
            self.wavelengths = np.array(self.wavelengths)
            
            # 提取5个LED通道的SPD数据（第2-6列）
            self.led_channels = {}
            channel_names = ['蓝光', '绿光', '红光', '暖白光(WW)', '冷白光(CW)']
            column_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
            
            for i, (channel_name, col_name) in enumerate(zip(channel_names, column_names)):
                spd_data = df[col_name].iloc[:len(self.wavelengths)].values
                # 确保数据为浮点数
                spd_data = pd.to_numeric(spd_data, errors='coerce')
                spd_data = np.nan_to_num(spd_data, 0)  # 将NaN替换为0
                self.led_channels[channel_name] = spd_data
            
            print(f"Successfully loaded {len(self.led_channels)} LED channels")
            print(f"Wavelength range: {self.wavelengths[0]:.0f}-{self.wavelengths[-1]:.0f} nm")
            print(f"Number of wavelength points: {len(self.wavelengths)}")
            
        except Exception as e:
            print(f"Error loading LED data: {e}")
            raise
    
    def load_solar_data(self, solar_data_path):
        """加载太阳光谱时间序列数据"""
        print("Loading solar spectrum time series data...")
        try:
            df = pd.read_excel(solar_data_path, sheet_name=0, header=0)
            
            # 检查数据结构
            print(f"Solar data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # 提取波长信息（第一列）
            wavelength_col = df.iloc[:, 0]
            solar_wavelengths = []
            
            # 从波长列提取数值（与LED数据处理类似）
            for i, wl_str in enumerate(wavelength_col):
                if pd.isna(wl_str):
                    break
                try:
                    # 处理格式如 "380(mW/m2/nm)" 的情况
                    if isinstance(wl_str, str):
                        wl_clean = wl_str.split('(')[0].strip()
                        wl = float(wl_clean)
                    elif isinstance(wl_str, (int, float)):
                        wl = float(wl_str)
                    else:
                        continue
                    
                    if 380 <= wl <= 780:  # 可见光范围
                        solar_wavelengths.append(wl)
                    else:
                        break
                except Exception as e:
                    print(f"Error parsing solar wavelength at row {i}: {wl_str}")
                    break
            
            solar_wavelengths = np.array(solar_wavelengths)
            print(f"Solar wavelength range: {solar_wavelengths[0]:.0f}-{solar_wavelengths[-1]:.0f} nm")
            
            # 插值到LED波长网格
            self.solar_spectra = {}
            
            # 处理时间列（除第一列波长外的所有列）
            time_columns = df.columns[1:]
            
            for col in time_columns:
                try:
                    # 处理时间标签
                    if hasattr(col, 'strftime'):  # datetime.time对象
                        time_label = col.strftime('%H:%M')
                    else:
                        time_label = str(col)
                    
                    # 提取该时间点的光谱数据
                    spectrum_data = df[col].iloc[:len(solar_wavelengths)].values
                    spectrum_data = pd.to_numeric(spectrum_data, errors='coerce')
                    
                    # 去除NaN值
                    valid_mask = ~np.isnan(spectrum_data) & ~np.isnan(solar_wavelengths)
                    if np.sum(valid_mask) < 10:  # 数据点太少，跳过
                        print(f"Skipping {time_label} due to insufficient data")
                        continue
                        
                    valid_wavelengths = solar_wavelengths[valid_mask]
                    valid_spectrum = spectrum_data[valid_mask]
                    
                    # 插值到LED波长网格
                    if len(valid_wavelengths) > 1:
                        interp_func = interp1d(valid_wavelengths, valid_spectrum, 
                                             kind='linear', bounds_error=False, fill_value=0)
                        interpolated_spectrum = interp_func(self.wavelengths)
                        
                        # 归一化
                        if np.max(interpolated_spectrum) > 0:
                            interpolated_spectrum = interpolated_spectrum / np.max(interpolated_spectrum)
                        
                        self.solar_spectra[time_label] = interpolated_spectrum
                        print(f"Successfully loaded spectrum for {time_label}")
                    
                except Exception as e:
                    print(f"Error processing time column {col}: {e}")
                    continue
            
            print(f"Successfully loaded {len(self.solar_spectra)} time points")
            print(f"Time points: {list(self.solar_spectra.keys())}")
            
        except Exception as e:
            print(f"Error loading solar data: {e}")
            raise
    
    def _calculate_manual_mel_der(self, spd_values):
        """
        手动计算mel-DER值，用于备选计算
        """
        try:
            # 简化的mel-DER计算
            # 使用蓝光区域(440-490nm)作为mel-DER的主要贡献
            blue_range = (self.wavelengths >= 440) & (self.wavelengths <= 490)
            total_range = (self.wavelengths >= 380) & (self.wavelengths <= 780)
            
            if np.sum(blue_range) == 0 or np.sum(total_range) == 0:
                return 0.5
            
            blue_power = np.trapz(spd_values[blue_range], self.wavelengths[blue_range])
            total_power = np.trapz(spd_values[total_range], self.wavelengths[total_range])
            
            if total_power > 0:
                mel_der_approx = (blue_power / total_power) * 2.0  # 经验系数
                return max(0.1, min(2.0, mel_der_approx))  # 限制在合理范围内
            else:
                return 0.5
        except:
            return 0.5

    def synthesize_spectrum(self, weights):
        """
        合成光谱
        
        Parameters:
        weights: 5个通道的权重数组
        
        Returns:
        合成的光谱数组
        """
        synthesized = np.zeros_like(self.wavelengths)
        channel_names = ['蓝光', '绿光', '红光', '暖白光(WW)', '冷白光(CW)']
        
        for i, channel_name in enumerate(channel_names):
            synthesized += weights[i] * self.led_channels[channel_name]
        
        return synthesized
    
    def calculate_spectrum_error(self, weights, target_spectrum):
        """
        计算光谱匹配误差
        
        Parameters:
        weights: LED通道权重
        target_spectrum: 目标太阳光谱
        
        Returns:
        光谱误差（均方根误差）
        """
        synthesized = self.synthesize_spectrum(weights)
        error = np.sqrt(np.mean((target_spectrum - synthesized)**2))
        return error
    
    def calculate_parameter_errors(self, weights, target_params):
        """
        计算关键参数匹配误差
        
        Parameters:
        weights: LED通道权重
        target_params: 目标参数字典 {'CCT': value, 'mel_DER': value, 'Duv': value}
        
        Returns:
        参数匹配误差
        """
        synthesized = self.synthesize_spectrum(weights)
        
        try:
            # 计算合成光谱的参数
            synthetic_params = self.calculator.calculate_all_parameters(
                self.wavelengths, synthesized)
            
            cct_error = 0
            mel_der_error = 0
            duv_error = 0
            
            if 'CCT' in target_params and target_params['CCT'] > 0:
                cct_error = abs(target_params['CCT'] - synthetic_params['CCT']) / target_params['CCT']
            
            if 'mel-DER' in target_params and target_params['mel-DER'] > 0:
                mel_der_error = abs(target_params['mel-DER'] - synthetic_params['mel-DER']) / target_params['mel-DER']
            
            if 'Duv' in target_params:
                duv_error = abs(target_params['Duv'] - synthetic_params['Duv']) / abs(target_params['Duv'])
            
            return cct_error + mel_der_error + duv_error * 0
            
        except Exception as e:
            print(f"Error calculating parameters: {e}")
            return float('inf')
    
    def optimize_single_timepoint(self, target_spectrum, time_label, 
                                 alpha=1.0, beta=0.5, gamma=0.3):
        """
        优化单个时间点的LED权重
        
        Parameters:
        target_spectrum: 目标太阳光谱
        time_label: 时间标签
        alpha, beta, gamma: 权重系数
        
        Returns:
        优化结果字典
        """
        print(f"Optimizing for time point: {time_label}")
        
        # 计算目标光谱的参数
        try:
            # 确保光谱数据为正值且归一化
            normalized_spectrum = np.maximum(target_spectrum, 0)
            if np.max(normalized_spectrum) > 0:
                normalized_spectrum = normalized_spectrum / np.max(normalized_spectrum)
            
            target_params = self.calculator.calculate_all_parameters(
                self.wavelengths, normalized_spectrum)
            
            # 检查mel-DER计算结果
            mel_der_value = target_params.get('mel-DER', 0)  # 注意是'mel-DER'不是'mel_DER'
            if mel_der_value == 0:
                print(f"警告: 时间点 {time_label} 的mel-DER为0，可能计算有误")
                # 手动计算mel-DER作为备选
                try:
                    manual_mel_der = self._calculate_manual_mel_der(normalized_spectrum)
                    if manual_mel_der > 0:
                        target_params['mel-DER'] = manual_mel_der
                        print(f"使用手动计算的mel-DER: {manual_mel_der:.4f}")
                except:
                    target_params['mel-DER'] = 0.5  # 使用默认值
            
            print(f"目标参数计算成功: CCT={target_params.get('CCT', 0):.0f}K, Duv={target_params.get('Duv', 0):.4f}, mel-DER={target_params.get('mel-DER', 0):.4f}")
        except Exception as e:
            print(f"计算目标参数时出错: {e}")
            # 使用默认值
            target_params = {'CCT': 5000, 'Duv': 0.0, 'Rf': 80, 'Rg': 95, 'mel-DER': 0.5}
        
        def objective(weights):
            """多目标优化目标函数"""
            # 光谱匹配误差
            spectrum_error = self.calculate_spectrum_error(weights, target_spectrum)
            
            # 参数匹配误差
            param_error = self.calculate_parameter_errors(weights, target_params)
            
            # 组合误差
            total_error = alpha * spectrum_error + beta * param_error
            
            return total_error
        
        # 约束条件：权重和为1
        def weight_constraint(weights):
            return np.sum(weights) - 1.0
        
        # 优化设置
        constraints = [{'type': 'eq', 'fun': weight_constraint}]
        bounds = [(0, 1) for _ in range(5)]  # 权重范围[0,1]
        
        # 多次随机初始化，选择最优结果
        best_result = None
        best_error = float('inf')

        for _ in range(10):  # 10次随机初始化
            # 随机初始权重（归一化）
            initial_weights = np.random.rand(5)
            initial_weights = initial_weights / np.sum(initial_weights)
            
            try:
                result = minimize(objective, initial_weights, 
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints,
                                options={'maxiter': 1000, 'ftol': 1e-9})
                
                if result.success and result.fun < best_error:
                    best_error = result.fun
                    best_result = result
                    
            except Exception as e:
                continue
        
        if best_result is None:
            print(f"时间点 {time_label} 优化失败，使用默认权重")
            # 返回均匀权重作为fallback
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            weights = best_result.x
            # 确保权重归一化
            weights = weights / np.sum(weights)
            print(f"时间点 {time_label} 优化成功，权重: [{', '.join([f'{w:.3f}' for w in weights])}]")
        
        # 计算结果
        synthesized_spectrum = self.synthesize_spectrum(weights)
        spectrum_rmse = np.sqrt(np.mean((target_spectrum - synthesized_spectrum)**2))
        
        try:
            synthetic_params = self.calculator.calculate_all_parameters(
                self.wavelengths, synthesized_spectrum)
            print(f"合成参数计算成功: CCT={synthetic_params.get('CCT', 0):.0f}K, Duv={synthetic_params.get('Duv', 0):.4f}, mel-DER={synthetic_params.get('mel-DER', 0):.4f}")
        except Exception as e:
            print(f"计算合成参数时出错: {e}")
            synthetic_params = {'CCT': 0, 'Duv': 0, 'Rf': 0, 'Rg': 0, 'mel-DER': 0}
        
        # 计算相关系数
        correlation = np.corrcoef(target_spectrum, synthesized_spectrum)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        return {
            'time': time_label,
            'weights': weights,
            'target_spectrum': target_spectrum,
            'synthesized_spectrum': synthesized_spectrum,
            'target_params': target_params,
            'synthetic_params': synthetic_params,
            'spectrum_rmse': spectrum_rmse,
            'correlation': correlation,
            'optimization_error': best_error
        }
    
    def generate_control_sequence(self):
        """生成全天控制序列"""
        print("正在为所有时间点生成控制序列...")
        
        control_sequence = []
        
        for time_label, target_spectrum in tqdm(self.solar_spectra.items(), 
                                              desc="优化时间点"):
            result = self.optimize_single_timepoint(target_spectrum, time_label)
            control_sequence.append(result)
            self.optimization_results[time_label] = result
        
        # 按时间排序
        try:
            control_sequence.sort(key=lambda x: x['time'])
        except:
            pass  # 如果时间格式无法排序，保持原顺序
        
        print(f"已为 {len(control_sequence)} 个时间点生成控制序列")
        return control_sequence
    
    def analyze_representative_timepoints(self, control_sequence):
        """分析三个代表性时间点"""
        print("Analyzing representative time points...")
        
        # 选择代表性时间点（早晨、正午、傍晚）
        if len(control_sequence) >= 3:
            # 均匀选择三个时间点
            indices = [0, len(control_sequence)//2, len(control_sequence)-1]
            representative_points = [control_sequence[i] for i in indices]
            labels = ['早晨', '正午', '傍晚']
        else:
            representative_points = control_sequence
            labels = [f'时间点{i+1}' for i in range(len(representative_points))]
        
        # 创建图表布局: 2列×3行 (光谱对比图在第一行，参数对比图在第二行)
        fig = plt.figure(figsize=(20, 12))
        
        for i, (result, label) in enumerate(zip(representative_points, labels)):
            if i >= 3:  # 最多显示3个时间点
                break
                
            # 光谱对比图 (第一行)
            ax_spectrum = plt.subplot(2, 3, i+1)
            ax_spectrum.plot(self.wavelengths, result['target_spectrum'], 
                    'b-', linewidth=2, label='目标太阳光谱')
            ax_spectrum.plot(self.wavelengths, result['synthesized_spectrum'], 
                    'r--', linewidth=2, label='LED合成光谱')
            ax_spectrum.set_xlabel('波长 (nm)')
            ax_spectrum.set_ylabel('相对功率')
            ax_spectrum.set_title(f'{label} ({result["time"]})\n光谱对比')
            ax_spectrum.legend()
            ax_spectrum.grid(True, alpha=0.3)
            
            # 参数对比图 (第二行) - 使用双纵坐标
            ax_params = plt.subplot(2, 3, i+4)
            target_params = result['target_params']
            synthetic_params = result['synthetic_params']
            
            # 准备所有参数数据
            cct_target = target_params.get('CCT', 0)
            cct_synthetic = synthetic_params.get('CCT', 0)
            duv_target = target_params.get('Duv', 0) * 100
            duv_synthetic = synthetic_params.get('Duv', 0) * 100
            mel_target = target_params.get('mel-DER', 0) * 10
            mel_synthetic = synthetic_params.get('mel-DER', 0) * 10
            rf_target = target_params.get('Rf', 0)
            rf_synthetic = synthetic_params.get('Rf', 0)
            rg_target = target_params.get('Rg', 0)
            rg_synthetic = synthetic_params.get('Rg', 0)
            
            # 设置参数名称和位置
            param_names = ['CCT', 'Duv', 'mel-DER', 'Rf', 'Rg']
            x_pos = np.arange(len(param_names))
            width = 0.35
            
            # CCT数据（使用左侧纵坐标）
            cct_target_vals = [cct_target, 0, 0, 0, 0]
            cct_synthetic_vals = [cct_synthetic, 0, 0, 0, 0]
            
            # 其他参数数据（使用右侧纵坐标）
            other_target_vals = [0, duv_target, mel_target, rf_target, rg_target]
            other_synthetic_vals = [0, duv_synthetic, mel_synthetic, rf_synthetic, rg_synthetic]
            
            # 绘制CCT柱状图（左侧纵坐标）
            bars_cct_target = ax_params.bar([0 - width/2], [cct_target], width, 
                                           label='CCT目标值', alpha=0.7, color='lightblue')
            bars_cct_synthetic = ax_params.bar([0 + width/2], [cct_synthetic], width, 
                                             label='CCT合成值', alpha=0.7, color='lightcoral')
            
            # 创建右侧纵坐标轴
            ax_params_right = ax_params.twinx()
            
            # 绘制其他参数柱状图（右侧纵坐标）
            other_x_pos = x_pos[1:]  # 除了CCT的其他位置
            other_target_data = [duv_target, mel_target, rf_target, rg_target]
            other_synthetic_data = [duv_synthetic, mel_synthetic, rf_synthetic, rg_synthetic]
            
            bars_other_target = ax_params_right.bar(other_x_pos - width/2, other_target_data, width,
                                                   label='其他参数目标值', alpha=0.7, color='lightgreen')
            bars_other_synthetic = ax_params_right.bar(other_x_pos + width/2, other_synthetic_data, width,
                                                     label='其他参数合成值', alpha=0.7, color='gold')
            
            # 添加数值标签
            # CCT标签
            for bar, value in zip(bars_cct_target, [cct_target]):
                height = bar.get_height()
                ax_params.text(bar.get_x() + bar.get_width()/2., height + cct_target * 0.02,
                              f'{value:.0f}K', ha='center', va='bottom', fontsize=9)
            
            for bar, value in zip(bars_cct_synthetic, [cct_synthetic]):
                height = bar.get_height()
                ax_params.text(bar.get_x() + bar.get_width()/2., height + cct_synthetic * 0.02,
                              f'{value:.0f}K', ha='center', va='bottom', fontsize=9)
            
            # 其他参数标签
            for bar, value, param in zip(bars_other_target, other_target_data, param_names[1:]):
                height = bar.get_height()
                if param == 'Duv':
                    label_text = f'{value:.4f}'
                elif param == 'mel-DER':
                    label_text = f'{value:.3f}'
                else:  # Rf, Rg
                    label_text = f'{value:.1f}'
                ax_params_right.text(bar.get_x() + bar.get_width()/2., height + max(other_target_data) * 0.02,
                                   label_text, ha='center', va='bottom', fontsize=9)
            
            for bar, value, param in zip(bars_other_synthetic, other_synthetic_data, param_names[1:]):
                height = bar.get_height()
                if param == 'Duv':
                    label_text = f'{value:.4f}'
                elif param == 'mel-DER':
                    label_text = f'{value:.3f}'
                else:  # Rf, Rg
                    label_text = f'{value:.1f}'
                ax_params_right.text(bar.get_x() + bar.get_width()/2., height + max(other_synthetic_data) * 0.02,
                                   label_text, ha='center', va='bottom', fontsize=9)
            
            # 设置坐标轴标签和标题
            ax_params.set_xlabel('参数类型')
            ax_params.set_ylabel('CCT (K)', color='red')
            ax_params_right.set_ylabel('其他参数数值', color='blue')
            ax_params.set_title(f'{label} 全参数对比')
            
            # 设置x轴刻度
            ax_params.set_xticks(x_pos)
            ax_params.set_xticklabels(param_names)
            
            # 设置图例
            lines1, labels1 = ax_params.get_legend_handles_labels()
            lines2, labels2 = ax_params_right.get_legend_handles_labels()
            ax_params.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax_params.grid(True, alpha=0.3)
            
            # 在光谱图上添加详细信息
            info_text = f"权重: B={result['weights'][0]:.3f}, G={result['weights'][1]:.3f}, R={result['weights'][2]:.3f}, WW={result['weights'][3]:.3f}, CW={result['weights'][4]:.3f}\n"
            info_text += f"RMSE: {result['spectrum_rmse']:.4f}, 相关系数: {result['correlation']:.4f}"
            ax_spectrum.text(0.02, 0.02, info_text, transform=ax_spectrum.transAxes, 
                           verticalalignment='bottom', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('Pictures/representative_timepoints_analysis.svg', format='svg', bbox_inches='tight')
        plt.show()
        
        return representative_points
    
    def save_results(self, control_sequence):
        """保存优化结果"""
        print("正在保存优化结果...")
        
        # 准备数据表
        results_data = []
        
        for result in control_sequence:
            row = {
                '时间': result['time'],
                '蓝光权重': result['weights'][0],
                '绿光权重': result['weights'][1], 
                '红光权重': result['weights'][2],
                '暖白光权重': result['weights'][3],
                '冷白光权重': result['weights'][4],
                '目标色温CCT': result['target_params'].get('CCT', 0),
                '合成色温CCT': result['synthetic_params'].get('CCT', 0),
                '目标Duv': result['target_params'].get('Duv', 0),
                '合成Duv': result['synthetic_params'].get('Duv', 0),
                '目标mel_DER': result['target_params'].get('mel-DER', 0),
                '合成mel_DER': result['synthetic_params'].get('mel-DER', 0),
                '光谱均方根误差': result['spectrum_rmse'],
                '相关系数': result['correlation']
            }
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv('DataFrames/problem3_optimization_results.csv', index=False, encoding='utf-8-sig')
        
        print("结果已保存到 DataFrames/problem3_optimization_results.csv")
    
    def plot_control_sequence(self, control_sequence):
        """绘制全天控制序列"""
        print("正在绘制控制序列...")
        
        times = [result['time'] for result in control_sequence]
        weights_matrix = np.array([result['weights'] for result in control_sequence])
        
        channel_names = ['蓝光', '绿光', '红光', '暖白光(WW)', '冷白光(CW)']
        colors = ['blue', 'green', 'red', 'orange', 'lightblue']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 权重变化图
        for i, (channel, color) in enumerate(zip(channel_names, colors)):
            ax1.plot(range(len(times)), weights_matrix[:, i], 
                    'o-', color=color, label=channel, linewidth=2, markersize=6)
        
        ax1.set_xlabel('时间点索引')
        ax1.set_ylabel('权重')
        ax1.set_title('全天LED通道权重变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 关键参数变化图
        cct_values = [result['synthetic_params'].get('CCT', 0) for result in control_sequence]
        mel_der_values = [result['synthetic_params'].get('mel-DER', 0) for result in control_sequence]
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(range(len(times)), cct_values, 'ro-', label='CCT', linewidth=2)
        line2 = ax2_twin.plot(range(len(times)), mel_der_values, 'bs-', label='mel-DER', linewidth=2)
        
        ax2.set_xlabel('时间点索引')
        ax2.set_ylabel('CCT (K)', color='red')
        ax2_twin.set_ylabel('mel-DER', color='blue')
        ax2.set_title('全天关键参数变化')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Pictures/control_sequence_analysis.svg', format='svg', bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("=== 问题三：全天候太阳光模拟LED控制策略 ===")
    
    try:
        # 初始化
        mimicry = SolarSpectrumMimicry()
        
        # 生成控制序列
        control_sequence = mimicry.generate_control_sequence()
        
        # 分析代表性时间点
        representative_points = mimicry.analyze_representative_timepoints(control_sequence)
        
        # 绘制控制序列
        mimicry.plot_control_sequence(control_sequence)
        
        # 保存结果
        mimicry.save_results(control_sequence)
        
        # 输出总结
        print("\n=== 优化结果总结 ===")
        print(f"处理了 {len(control_sequence)} 个时间点")
        
        if control_sequence:
            avg_rmse = np.mean([r['spectrum_rmse'] for r in control_sequence])
            avg_correlation = np.mean([r['correlation'] for r in control_sequence])
            
            print(f"平均光谱RMSE: {avg_rmse:.4f}")
            print(f"平均相关系数: {avg_correlation:.4f}")
            
            print("\n代表性时间点分析:")
            for i, point in enumerate(representative_points):
                print(f"时间点 {i+1} ({point['time']}):")
                print(f"  权重: [{', '.join([f'{w:.3f}' for w in point['weights']])}]")
                print(f"  RMSE: {point['spectrum_rmse']:.4f}")
                print(f"  相关系数: {point['correlation']:.4f}")
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
