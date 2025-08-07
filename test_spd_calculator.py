"""
SPD参数计算器使用示例和测试
"""

from spd_calculator import SPDCalculator, load_spd_data
import os

def test_with_real_data():
    """使用真实数据测试计算器"""
    print("=== 使用真实SPD数据测试 ===")
    
    # 创建计算器实例
    calculator = SPDCalculator()
    
    # 测试不同的数据文件
    test_files = [
        'C/附录1.xlsx',
        # 可以添加更多测试文件
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n--- 处理文件: {file_path} ---")
            
            # 加载SPD数据
            wavelengths, spd_values = load_spd_data(file_path)
            
            if wavelengths is not None and spd_values is not None:
                # 计算所有参数
                results = calculator.calculate_all_parameters(wavelengths, spd_values)
                
                # 打印结果
                calculator.print_results(results)
                
                # 保存结果到字典以便进一步分析
                return results
            else:
                print(f"无法读取文件: {file_path}")
        else:
            print(f"文件不存在: {file_path}")

def analyze_light_quality(results):
    """分析光源质量"""
    if not results:
        return
    
    print("\n=== 光源质量分析 ===")
    
    cct = results.get('CCT', 0)
    duv = results.get('Duv', 0)
    rf = results.get('Rf', 0)
    rg = results.get('Rg', 0)
    mel_der = results.get('mel-DER', 0)
    
    # 色温分析
    if cct < 3000:
        temp_desc = "暖白光 (适合放松环境)"
    elif cct < 4000:
        temp_desc = "中性偏暖 (适合居住空间)"
    elif cct < 5000:
        temp_desc = "中性白光 (适合办公环境)"
    elif cct < 6500:
        temp_desc = "冷白光 (适合工作环境)"
    else:
        temp_desc = "日光色 (适合精密作业)"
    
    print(f"色温分析: {cct:.0f}K - {temp_desc}")
    
    # Duv分析
    if abs(duv) < 0.002:
        duv_desc = "色品质优秀"
    elif abs(duv) < 0.005:
        duv_desc = "色品质良好"
    elif abs(duv) < 0.010:
        duv_desc = "色品质一般"
    else:
        duv_desc = "色品质偏差较大"
    
    print(f"色品偏差: Duv={duv:.4f} - {duv_desc}")
    
    # 显色性分析
    if rf >= 90:
        rf_desc = "显色性优秀"
    elif rf >= 80:
        rf_desc = "显色性良好"
    elif rf >= 70:
        rf_desc = "显色性一般"
    else:
        rf_desc = "显色性较差"
    
    print(f"显色性分析: Rf={rf:.1f} - {rf_desc}")
    
    # 生理节律效应分析
    if mel_der > 1.2:
        mel_desc = "强激活效应 (适合白天使用)"
    elif mel_der > 0.8:
        mel_desc = "中等激活效应 (适合日常照明)"
    elif mel_der > 0.4:
        mel_desc = "低激活效应 (适合晚间使用)"
    else:
        mel_desc = "微弱激活效应 (适合睡前使用)"
    
    print(f"生理节律: mel-DER={mel_der:.3f} - {mel_desc}")
    
    # 应用建议
    print("\n=== 应用建议 ===")
    if 2700 <= cct <= 3500 and mel_der < 0.6:
        print("✓ 适合用于睡前照明和放松环境")
    
    if 4000 <= cct <= 6500 and rf >= 80:
        print("✓ 适合用于日常办公和阅读照明")
    
    if mel_der > 1.0 and rf >= 85:
        print("✓ 适合用于提高警觉性的工作环境")
    
    if abs(duv) < 0.005 and rf >= 90:
        print("✓ 适合用于对色彩要求较高的场所")

def batch_analysis(file_list):
    """批量分析多个SPD文件"""
    print("=== 批量SPD文件分析 ===")
    
    calculator = SPDCalculator()
    results_list = []
    
    for file_path in file_list:
        if os.path.exists(file_path):
            print(f"\n--- 分析文件: {file_path} ---")
            wavelengths, spd_values = load_spd_data(file_path)
            
            if wavelengths is not None:
                results = calculator.calculate_all_parameters(wavelengths, spd_values)
                results['file_path'] = file_path
                results_list.append(results)
                
                # 简要显示结果
                print(f"CCT: {results['CCT']:.0f}K, Rf: {results['Rf']:.1f}, mel-DER: {results['mel-DER']:.3f}")
            
    return results_list

def create_summary_report(results_list):
    """创建汇总报告"""
    if not results_list:
        return
    
    print("\n=== SPD分析汇总报告 ===")
    print("文件名\t\tCCT(K)\tDuv\tRf\tRg\tmel-DER")
    print("-" * 70)
    
    for results in results_list:
        file_name = os.path.basename(results.get('file_path', 'Unknown'))
        print(f"{file_name[:15]:<15}\t{results['CCT']:.0f}\t{results['Duv']:.4f}\t"
              f"{results['Rf']:.1f}\t{results['Rg']:.1f}\t{results['mel-DER']:.3f}")

if __name__ == "__main__":
    # 测试单个文件
    results = test_with_real_data()
    
    if results:
        # 分析光源质量
        analyze_light_quality(results)
        
        # 如果有多个文件，可以进行批量分析
        # 示例：批量分析不同的SPD文件
        """
        file_list = [
            'C/附录1.xlsx',
            'C/附录2_LED_SPD.xlsx',
            # 添加更多文件路径
        ]
        
        batch_results = batch_analysis(file_list)
        create_summary_report(batch_results)
        """
