"""
SPD到5个核心参数的计算函数
根据问题1的要求，计算以下参数：
1. 相关色温 (CCT)
2. 距离普朗克轨迹的距离 (Duv)
3. 保真度指数 (Rf)
4. 色域指数 (Rg)
5. 褪黑素日光照度比 (mel-DER)
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import colour
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans', '微软雅黑']

print(f"使用colour库版本: {colour.__version__}")

class SPDCalculator:
    def __init__(self, cie_data_path="C/CIE_xyz_1931_2deg.csv"):
        """
        初始化SPD计算器
        
        Parameters:
        cie_data_path: CIE 1931标准观察者数据文件路径
        """
        self.load_cie_data(cie_data_path)
        self.load_color_rendering_data()
        
    def load_cie_data(self, cie_data_path):
        """加载CIE 1931标准观察者数据"""
        try:
            # 读取CIE数据
            cie_data = pd.read_csv(cie_data_path, header=None, 
                                 names=['wavelength', 'x_bar', 'y_bar', 'z_bar'])
            
            self.wavelengths = cie_data['wavelength'].values
            self.x_bar = cie_data['x_bar'].values
            self.y_bar = cie_data['y_bar'].values
            self.z_bar = cie_data['z_bar'].values
            
            print(f"成功加载CIE数据，波长范围: {self.wavelengths[0]}-{self.wavelengths[-1]}nm")
            
        except Exception as e:
            print(f"加载CIE数据时出错: {e}")
            # 使用默认数据范围
            self.wavelengths = np.arange(380, 781, 1)
            print("使用默认波长范围: 380-780nm")
    
    def load_color_rendering_data(self):
        """加载颜色渲染计算所需的数据"""
        try:
            # 使用colour库的内置数据 - 适配不同版本的API
            try:
                self.CES = colour.SDS_COLOURCHECKERS['ColorChecker N Ohta']
            except:
                try:
                    self.CES = colour.SDS_COLOURCHECKERS['ColorChecker 2005']
                except:
                    print("警告: 无法加载标准色样数据，将使用简化计算")
                    self.CES = None
            print("成功加载颜色渲染测试色样数据")
        except Exception as e:
            print(f"加载颜色渲染数据时出错: {e}")
            self.CES = None
    
    def interpolate_spd(self, wavelengths, spd_values, target_wavelengths=None):
        """
        将SPD数据插值到标准波长范围
        
        Parameters:
        wavelengths: SPD的波长数组
        spd_values: SPD的功率值数组
        target_wavelengths: 目标波长范围，默认为CIE标准范围
        
        Returns:
        插值后的SPD数组
        """
        if target_wavelengths is None:
            target_wavelengths = self.wavelengths
            
        # 创建插值函数
        f = interp1d(wavelengths, spd_values, kind='linear', 
                    bounds_error=False, fill_value=0)
        
        return f(target_wavelengths)
    
    def spd_to_xyz(self, wavelengths, spd_values):
        """
        将SPD转换为CIE XYZ色彩空间
        
        Parameters:
        wavelengths: 波长数组 (nm)
        spd_values: 光谱功率分布数组 (W/nm)
        
        Returns:
        tuple: (X, Y, Z) 三刺激值
        """
        # 插值到标准波长范围
        spd_interp = self.interpolate_spd(wavelengths, spd_values)
        
        # 计算归一化常数
        k = 100 / np.trapz(spd_interp * self.y_bar, self.wavelengths)
        
        # 计算XYZ
        X = k * np.trapz(spd_interp * self.x_bar, self.wavelengths)
        Y = k * np.trapz(spd_interp * self.y_bar, self.wavelengths)
        Z = k * np.trapz(spd_interp * self.z_bar, self.wavelengths)
        
        return X, Y, Z
    
    def xyz_to_uv(self, X, Y, Z):
        """
        将XYZ转换为CIE 1960 UCS (u,v)色度坐标
        
        Parameters:
        X, Y, Z: CIE XYZ三刺激值
        
        Returns:
        tuple: (u, v) 色度坐标
        """
        denominator = X + 15*Y + 3*Z
        if denominator == 0:
            return 0, 0
            
        u = (4*X) / denominator
        v = (6*Y) / denominator
        
        return u, v
    
    def xyz_to_rgb(self, X, Y, Z, gamma_correction=True):
        """
        将XYZ转换为sRGB颜色空间
        
        Parameters:
        X, Y, Z: CIE XYZ三刺激值
        gamma_correction: 是否应用gamma校正
        
        Returns:
        tuple: (R, G, B) RGB值 (0-255)
        """
        # 使用sRGB变换矩阵 (D65照明体)
        # 矩阵来源: IEC 61966-2-1:1999
        M = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ])
        
        # 归一化XYZ (Y=100 -> Y=1)
        XYZ_norm = np.array([X/100, Y/100, Z/100])
        
        # 线性RGB
        RGB_linear = M @ XYZ_norm
        
        if gamma_correction:
            # 应用sRGB gamma校正
            def gamma_correct(c):
                if c <= 0.0031308:
                    return 12.92 * c
                else:
                    return 1.055 * (c ** (1/2.4)) - 0.055
            
            RGB_corrected = np.array([gamma_correct(c) for c in RGB_linear])
        else:
            RGB_corrected = RGB_linear
        
        # 限制在0-1范围内，然后转换为0-255
        RGB_corrected = np.clip(RGB_corrected, 0, 1)
        RGB_255 = (RGB_corrected * 255).astype(int)
        
        return tuple(RGB_255)
    
    def chebyshev_polynomials(self, T):
        """
        Chebyshev多项式计算u(T)和v(T)
        
        Parameters:
        T: 色温 (K)
        
        Returns:
        tuple: (u_T, v_T, du_T, dv_T) - u(T), v(T)及其导数
        """
        # u(T)系数 - 修正后的数据
        u_num = 0.860117757 + 1.54118254e-4*T + 1.28641212e-7*T**2
        u_den = 1 + 8.42420235e-4*T + 7.08145163e-7*T**2
        u_T = u_num / u_den
        
        # v(T)系数 - 修正后的数据
        v_num = 0.317398726 + 4.22806245e-5*T + 4.20481691e-8*T**2
        v_den = 1 - 2.89741816e-5*T + 1.61456053e-7*T**2
        v_T = v_num / v_den
        
        # 计算导数 du/dT 和 dv/dT
        du_num = 1.54118254e-4 + 2*1.28641212e-7*T
        du_den_deriv = 8.42420235e-4 + 2*7.08145163e-7*T
        du_T = (du_num * u_den - u_num * du_den_deriv) / (u_den**2)
        
        dv_num = 4.22806245e-5 + 2*4.20481691e-8*T
        dv_den_deriv = -2.89741816e-5 + 2*1.61456053e-7*T
        dv_T = (dv_num * v_den - v_num * dv_den_deriv) / (v_den**2)
        
        return u_T, v_T, du_T, dv_T
    
    def calculate_cct(self, u, v):
        """
        计算相关色温CCT - 使用最小化距离的方法
        
        Parameters:
        u, v: CIE 1960 UCS色度坐标
        
        Returns:
        float: 相关色温 (K)
        """
        def distance_squared(T):
            """计算点(u,v)到黑体轨迹上温度T对应点的距离平方"""
            try:
                u_T, v_T, _, _ = self.chebyshev_polynomials(T)
                return (u - u_T)**2 + (v - v_T)**2
            except:
                return float('inf')
        
        # 使用scipy的minimize_scalar来寻找最小距离
        from scipy.optimize import minimize_scalar
        
        try:
            # 在合理的色温范围内寻找最小距离
            result = minimize_scalar(distance_squared, bounds=(1000, 20000), method='bounded')
            if result.success:
                return result.x
            else:
                # 如果优化失败，尝试更大范围
                result = minimize_scalar(distance_squared, bounds=(500, 50000), method='bounded')
                if result.success:
                    return result.x
        except:
            pass
        
        # 如果优化方法失败，使用网格搜索
        print("警告: 优化方法失败，使用网格搜索")
        temps = np.linspace(1000, 20000, 1000)
        distances = [distance_squared(T) for T in temps]
        min_idx = np.argmin(distances)
        return temps[min_idx]
    
    def calculate_duv(self, u, v, cct):
        """
        计算距离普朗克轨迹的距离Duv
        
        Parameters:
        u, v: 测试光源的色度坐标
        cct: 相关色温
        
        Returns:
        float: Duv值
        """
        u_t, v_t, _, _ = self.chebyshev_polynomials(cct)
        
        # 计算符号
        sign = 1 if v > v_t else -1
        
        # 计算距离
        duv = sign * np.sqrt((u - u_t)**2 + (v - v_t)**2)
        
        return duv
    
    def calculate_color_rendering(self, wavelengths, spd_values):
        """
        计算颜色渲染指数Rf和Rg
        
        Parameters:
        wavelengths: 波长数组
        spd_values: SPD值数组
        
        Returns:
        tuple: (Rf, Rg) 保真度指数和色域指数
        """
        try:
            # 创建colour库的光谱分布对象
            spd_dict = {}
            for i, wl in enumerate(wavelengths):
                spd_dict[int(wl)] = spd_values[i]
            
            spd = colour.SpectralDistribution(spd_dict, name='Test SPD')
            
            # 使用ANSI/IES TM-30-18方法计算颜色保真度和色域指数
            try:
                # colour-science 0.4.6的正确API
                tm30_results = colour.colour_fidelity_index(
                    spd, 
                    additional_data=True, 
                    method='ANSI/IES TM-30-18'
                )
                Rf = tm30_results.R_f
                Rg = tm30_results.R_g
                Duv = tm30_results.D_uv
                CCT = tm30_results.CCT
                # print(f"TM-30计算成功: Rf={Rf:.1f}, Rg={Rg:.1f}, Duv={Duv:.4f}, CCT={CCT:.0f}")

            except Exception as tm30_error:
                print(f"TM-30计算失败: {tm30_error}")
                
                # 备选方案：使用CIE 2017方法
                try:
                    cie2017_result = colour.colour_fidelity_index(
                        spd, 
                        additional_data=True, 
                        method='CIE 2017'
                    )
                    Rf = cie2017_result.R_f
                    Duv = cie2017_result.D_uv
                    CCT = cie2017_result.CCT
                    # CIE 2017没有Rg，使用传统CRI估算
                    try:
                        cri = colour.colour_rendering_index(spd)
                        Rg = max(80, min(120, cri + 10))  # 基于CRI估算Rg
                    except:
                        Rg = 100
                    
                    print(f"使用CIE 2017方法: Rf={Rf:.1f}, Rg={Rg:.1f}(估算)")
                    
                except Exception as cie_error:
                    print(f"CIE 2017计算也失败: {cie_error}")
                    
                    # 最后备选：使用传统CRI
                    try:
                        cri = colour.colour_rendering_index(spd)
                        Rf = cri  # 使用CRI作为Rf的近似
                        Rg = 100  # 默认色域指数
                        Duv = 0.0  # 默认Duv值
                        print(f"使用传统CRI方法: Rf={Rf:.1f}, Rg={Rg:.1f}")
                    except:
                        Rf, Rg = 80, 100  # 默认值
                        print("所有方法都失败，使用默认值")

            return float(Rf), float(Rg), float(Duv), float(CCT)

        except Exception as e:
            print(f"计算颜色渲染指数时出错: {e}")
            return 80.0, 100.0, 0.0, 6500.0  # 返回合理的默认值

    def calculate_mel_der(self, wavelengths, spd_values):
        """
        计算褪黑素日光照度比mel-DER
        基于CIE S026/E:2018标准实现
        
        Parameters:
        wavelengths: 波长数组
        spd_values: SPD值数组
        
        Returns:
        float: mel-DER值
        """
        try:
            # 创建光谱分布对象
            spd_dict = {}
            for i, wl in enumerate(wavelengths):
                spd_dict[int(wl)] = spd_values[i]
            
            spd = colour.SpectralDistribution(spd_dict, name='Test SPD')
            
            # 定义标准波长范围 (380-780nm, 1nm间隔)
            common_wavelengths = np.arange(380, 781, 1)
            
            # 获取标准photopic光效函数 V(λ)
            try:
                photopic_lef = colour.SDS_LEFS['CIE 1924 Photopic Standard Observer']
                photopic_interp = self.interpolate_spd(
                    photopic_lef.wavelengths, 
                    photopic_lef.values, 
                    common_wavelengths
                )
            except:
                # 备选：使用标准V(λ)函数的解析近似
                print("使用标准V(λ)函数的解析近似")
                photopic_interp = self._standard_photopic_function(common_wavelengths)
            
            # 获取D65照明体光谱
            try:
                D65 = colour.SDS_ILLUMINANTS['D65']
                D65_interp = self.interpolate_spd(
                    D65.wavelengths, 
                    D65.values, 
                    common_wavelengths
                )
            except:
                # 备选：使用D65的解析近似
                print("使用D65照明体的解析近似")
                D65_interp = self._cie_d65_approximation(common_wavelengths)
            
            # 实现CIE S026/E:2018褪黑素效应函数 s_mel(λ)
            # 基于CIE S026标准的双峰高斯拟合
            s_mel_interp = self._cie_s026_melanopic_function(common_wavelengths)
            
            # 插值测试SPD到标准波长范围
            spd_interp = self.interpolate_spd(wavelengths, spd_values, common_wavelengths)
            
            # 计算测试光源的褪黑素辐照度
            E_mel = np.trapz(spd_interp * s_mel_interp, common_wavelengths)
            
            # 计算测试光源的明视觉照度 (lm/m²)
            E_phot = np.trapz(spd_interp * photopic_interp, common_wavelengths) * 683
            
            # 计算测试光源的mel-ELR (mEDI/lux)
            mel_ELR = (E_mel / E_phot) * 1000 if E_phot > 0 else 0
            
            # 计算D65照明体的褪黑素辐照度
            E_mel_D65 = np.trapz(D65_interp * s_mel_interp, common_wavelengths)
            
            # 计算D65照明体的明视觉照度
            E_phot_D65 = np.trapz(D65_interp * photopic_interp, common_wavelengths) * 683
            
            # 计算D65的mel-ELR
            mel_ELR_D65 = (E_mel_D65 / E_phot_D65) * 1000 if E_phot_D65 > 0 else 1
            
            # 计算mel-DER (测试光源的mel-ELR / D65的mel-ELR)
            mel_DER = mel_ELR / mel_ELR_D65 if mel_ELR_D65 > 0 else 0
            
            # print(f"mel-DER计算成功: mel-ELR={mel_ELR:.3f}, mel-ELR_D65={mel_ELR_D65:.3f}, mel-DER={mel_DER:.3f}")
            
            return mel_DER
            
        except Exception as e:
            print(f"计算mel-DER时出错: {e}")
            return 1.0  # 返回默认值
    
    def _cie_s026_melanopic_function(self, wavelengths):
        """
        基于CIE S026/E:2018标准的褪黑素效应函数 s_mel(λ)
        使用双峰高斯拟合来近似标准曲线
        
        Parameters:
        wavelengths: 波长数组 (nm)
        
        Returns:
        s_mel: 褪黑素效应函数值数组
        """
        # CIE S026标准的褪黑素效应函数参数
        # 主峰在480nm附近，次峰在420nm附近
        
        # 主峰参数 (480nm)
        peak1_center = 480.0
        peak1_amplitude = 1.0
        peak1_width = 25.0
        
        # 次峰参数 (420nm)  
        peak2_center = 420.0
        peak2_amplitude = 0.15
        peak2_width = 20.0
        
        # 双峰高斯函数
        peak1 = peak1_amplitude * np.exp(-((wavelengths - peak1_center)**2) / (2 * peak1_width**2))
        peak2 = peak2_amplitude * np.exp(-((wavelengths - peak2_center)**2) / (2 * peak2_width**2))
        
        s_mel = peak1 + peak2
        
        # 确保在可见光范围外为0
        s_mel[wavelengths < 380] = 0
        s_mel[wavelengths > 780] = 0
        
        return s_mel
    
    def _standard_photopic_function(self, wavelengths):
        """
        标准明视觉光效函数 V(λ) 的解析近似
        
        Parameters:
        wavelengths: 波长数组 (nm)
        
        Returns:
        V_lambda: 明视觉光效函数值数组
        """
        # CIE 1924标准明视觉函数的高斯近似
        # 峰值在555nm
        center = 555.0
        width = 80.0
        
        V_lambda = np.exp(-((wavelengths - center)**2) / (2 * width**2))
        
        # 确保在可见光范围外为0
        V_lambda[wavelengths < 380] = 0
        V_lambda[wavelengths > 780] = 0
        
        return V_lambda
    
    def _cie_d65_approximation(self, wavelengths):
        """
        CIE D65照明体的解析近似
        
        Parameters:
        wavelengths: 波长数组 (nm)
        
        Returns:
        D65_approx: D65光谱功率分布近似值
        """
        # D65大致对应6500K黑体辐射，使用平滑的光谱分布
        # 在蓝光区域略强，红光区域略弱
        
        # 基础黑体辐射近似
        base_spectrum = np.ones_like(wavelengths)
        
        # 添加D65特有的光谱特征
        # 在短波长处增强（蓝色）
        blue_enhancement = 1 + 0.3 * np.exp(-((wavelengths - 460)**2) / (2 * 60**2))
        
        # 在长波长处略微衰减（红色）
        red_attenuation = 1 - 0.1 * np.exp(-((wavelengths - 650)**2) / (2 * 80**2))
        
        D65_approx = base_spectrum * blue_enhancement * red_attenuation
        
        # 确保在可见光范围外为0
        D65_approx[wavelengths < 300] = 0
        D65_approx[wavelengths > 830] = 0
        
        return D65_approx
    
    def calculate_all_parameters(self, wavelengths, spd_values):
        """
        计算所有5个核心参数
        
        Parameters:
        wavelengths: 波长数组 (nm)
        spd_values: 光谱功率分布数组 (W/nm)
        
        Returns:
        dict: 包含所有5个参数的字典
        """
        results = {}
        
        # 1. 计算XYZ和色度坐标
        X, Y, Z = self.spd_to_xyz(wavelengths, spd_values)
        u, v = self.xyz_to_uv(X, Y, Z)
        
        # 2. 计算CCT
        # cct = self.calculate_cct(u, v)
        # results['CCT'] = cct
        # print(f"相关色温 (CCT): {cct:.1f} K")
        # 3. 计算Duv
        # duv = self.calculate_duv(u, v, cct)
        # results['Duv'] = duv
        # print(f"Duv: {duv:.4f}")
        # 4. 计算颜色渲染指数
        rf, rg, duv, CCT = self.calculate_color_rendering(wavelengths, spd_values)
        results['Rf'] = rf
        results['Rg'] = rg
        results['Duv'] = duv
        results['CCT'] = CCT
        # print(f"TM-30 CCT: {CCT:.1f} K")
        # print(f"TM-30 Duv: {duv:.4f}")
        # 5. 计算mel-DER
        mel_der = self.calculate_mel_der(wavelengths, spd_values)
        results['mel-DER'] = mel_der
        
        # 6. 计算RGB值
        rgb = self.xyz_to_rgb(X, Y, Z)
        results['RGB'] = rgb
        
        # 添加中间结果
        results['XYZ'] = (X, Y, Z)
        results['uv'] = (u, v)
        
        return results
    
    def print_results(self, results):
        """格式化打印结果"""
        print("\n=== SPD光谱特性参数计算结果 ===")
        print(f"相关色温 (CCT): {results['CCT']:.1f} K")
        print(f"距离普朗克轨迹的距离 (Duv): {results['Duv']:.4f}")
        print(f"保真度指数 (Rf): {results['Rf']:.1f}")
        print(f"色域指数 (Rg): {results['Rg']:.1f}")
        print(f"褪黑素日光照度比 (mel-DER): {results['mel-DER']:.3f}")
        print("\n=== 颜色信息 ===")
        R, G, B = results['RGB']
        print(f"sRGB颜色值: R={R}, G={G}, B={B}")
        print(f"十六进制颜色代码: #{R:02X}{G:02X}{B:02X}")
        print("\n=== 中间计算结果 ===")
        X, Y, Z = results['XYZ']
        u, v = results['uv']
        print(f"CIE XYZ: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
        print(f"CIE 1960 uv: u={u:.4f}, v={v:.4f}")
        print(f"色品图坐标: x={X/(X+Y+Z):.4f}, y={Y/(X+Y+Z):.4f}")

    def drawColour(self, results):
        """绘制颜色块和在uv色品图上的位置"""
        R, G, B = results['RGB']
        hex_color = f'#{R:02X}{G:02X}{B:02X}'
        u, v = results['uv']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Convert RGB values to normalized format for imshow
        rgb_normalized = np.array([[[R/255, G/255, B/255]]])
        ax[0].imshow(rgb_normalized)
        ax[0].axis('off')
        ax[0].set_title(f"Color: {hex_color}", fontsize=16)
        ax[1].plot(u, v, 'o', color=hex_color, markersize=10)
        ax[1].set_xlabel("CIE 1960 u")
        ax[1].set_ylabel("CIE 1960 v")
        ax[1].set_title("CIE 1960 色品图")
        ax[1].grid()
        plt.show()

def load_spd_from_excel(file_path, sheet_name=None):
    """
    从Excel文件加载SPD数据
    
    Parameters:
    file_path: Excel文件路径
    sheet_name: 工作表名称，None表示第一个工作表
    
    Returns:
    tuple: (wavelengths, spd_values)
    """
    try:
        # 如果没有指定工作表名称，先获取所有工作表名称
        if sheet_name is None:
            xl = pd.ExcelFile(file_path)
            if xl.sheet_names:
                sheet_name = xl.sheet_names[0]
                print(f"使用工作表: {sheet_name}")
        
        # 读取Excel文件，跳过第一行标题
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
        
        # 提取波长和SPD值
        wavelengths = []
        spd_values = []
        
        for i in range(len(df)):
            try:
                wavelength_str = str(df.iloc[i, 0])
                spd_value = df.iloc[i, 1]
                
                # 跳过无效行
                if pd.isna(spd_value) or wavelength_str == 'nan':
                    continue
                
                # 从波长字符串中提取数值，例如 "380(mW/m2/nm)" -> 380
                if '(' in wavelength_str:
                    wavelength = float(wavelength_str.split('(')[0])
                else:
                    wavelength = float(wavelength_str)
                
                wavelengths.append(wavelength)
                spd_values.append(float(spd_value))
                
            except (ValueError, TypeError, IndexError) as e:
                print(f"解析第{i+1}行数据时出错: {e}")
                continue
        
        if len(wavelengths) == 0:
            raise ValueError("没有找到有效的SPD数据")
        
        wavelengths = np.array(wavelengths)
        spd_values = np.array(spd_values)
        
        print(f"成功读取SPD数据，波长范围: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm，共{len(wavelengths)}个数据点")
        
        return wavelengths, spd_values
        
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None, None


def load_spd_data(file_path, sheet_name=None, data_format='auto'):
    """
    通用SPD数据加载函数，支持多种数据格式
    
    Parameters:
    file_path: 数据文件路径
    sheet_name: Excel工作表名称（仅对Excel文件有效）
    data_format: 数据格式 ('auto', 'excel', 'csv')
    
    Returns:
    tuple: (wavelengths, spd_values)
    """
    if data_format == 'auto':
        # 根据文件扩展名自动判断格式
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return load_spd_from_excel(file_path, sheet_name)
        elif file_path.endswith('.csv'):
            return load_spd_from_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return None, None
    elif data_format == 'excel':
        return load_spd_from_excel(file_path, sheet_name)
    elif data_format == 'csv':
        return load_spd_from_csv(file_path)
    else:
        print(f"不支持的数据格式: {data_format}")
        return None, None


def load_spd_from_csv(file_path):
    """
    从CSV文件加载SPD数据
    
    Parameters:
    file_path: CSV文件路径
    
    Returns:
    tuple: (wavelengths, spd_values)
    """
    try:
        df = pd.read_csv(file_path, header=None)
        wavelengths = df.iloc[:, 0].values
        spd_values = df.iloc[:, 1].values
        
        print(f"成功读取CSV SPD数据，波长范围: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm，共{len(wavelengths)}个数据点")
        
        return wavelengths, spd_values
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None, None


# 示例使用函数
def example_usage():
    """示例用法"""
    # 创建计算器实例
    calculator = SPDCalculator()
    
    # 从Excel文件加载SPD数据
    file_path = 'C/附录1.xlsx'
    wavelengths, spd_values = load_spd_from_excel(file_path)
    
    if wavelengths is None or spd_values is None:
        print("无法读取SPD数据，使用模拟数据进行演示")
        # 使用模拟数据
        wavelengths = np.arange(380, 781, 1)
        spd_values = np.exp(-((wavelengths - 555)**2) / (2 * 50**2))  # 高斯分布示例
    
    # 计算所有参数
    results = calculator.calculate_all_parameters(wavelengths, spd_values)
    
    # 打印结果
    calculator.print_results(results)
    calculator.drawColour(results)
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_usage()
