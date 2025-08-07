# 需要计算5个核心参数

## 颜色特性

相关色温 $CCT$

$SPD \to CIE \space 1931 \space XYZ$

$$
\begin{aligned}
X &= k \sum_{\lambda=380}^{780} S(\lambda)\,\bar x(\lambda)\,\Delta\lambda \\
Y &= k \sum_{\lambda=380}^{780} S(\lambda)\,\bar y(\lambda)\,\Delta\lambda \\
Z &= k \sum_{\lambda=380}^{780} S(\lambda)\,\bar z(\lambda)\,\Delta\lambda
\end{aligned}
\quad\text{其中}\; k=\frac{100}{\sum S(\lambda)\bar y(\lambda)}
$$

$XYZ \to CIE \space 1960 \space UCS \space (u,v)$

$$
u=\frac{4X}{X+15Y+3Z},\quad v=\frac{6Y}{X+15Y+3Z}
$$

用 Chebyshev 多项式求 CCT

$$
\begin{aligned}
u(T) &=\frac{0.8600117777+1.5418255\times10^{-4}T+2.881221\times10^{-7}T^{2}}
            {1+8.4420225\times10^{-4}T+7.08145163\times10^{-7}T^{2}} \\[4pt]
v(T) &=\frac{0.317398726+4.228066255\times10^{-5}T+4.20181691\times10^{-8}T^{2}}
            {1-2.89741816\times10^{-5}T+1.615505\times10^{-7}T^{2}}
\end{aligned}
$$

构建单变量方程

$$
F(T)=\frac{u(T)-u}{v(T)-v}+ \frac{u'(T)}{v'(T)}=0
$$

```python
from scipy.optimize import brentq

def F(T):
    uT = u_poly(T)          # Chebyshev u(T)
    vT = v_poly(T)          # Chebyshev v(T)
    du = du_poly(T)         # u'(T)
    dv = dv_poly(T)         # v'(T)
    return (uT - u)/(vT - v) + du/dv

T_cct = brentq(F, 1000, 20000)   # 返回色温（K）
```

距离普朗克轨迹的距离 $Duv$

$$
Duv = \text{sign}(v_s - v_t) \cdot \sqrt{(u_s - u_t)^2 + (v_s - v_t)^2}
$$

## 颜色还原参数

保真度指数 $Rf$

SPD → XYZ

$$
\begin{aligned}
X_i = k \sum_{\lambda=380}^{780} S(\lambda)\,R_i(\lambda)\,\bar x_{10}(\lambda)\,\Delta\lambda
\\ 

Y_i = k \sum_{\lambda=380}^{780} S(\lambda)\,R_i(\lambda)\,\bar y_{10}(\lambda)\,\Delta\lambda
\\

Z_i = k \sum_{\lambda=380}^{780} S(\lambda)\,R_i(\lambda)\,\bar z_{10}(\lambda)\,\Delta\lambda
\end{aligned}
$$

$$
k = \frac{100}{\sum_{\lambda} S(\lambda)\,\bar y_{10}(\lambda)\,\Delta\lambda}
$$

1. $XYZ \to CIE \space 1976 \space LAB \space D65$

$$
L^* = 116 f\left(\frac{Y}{Y_n}\right) - 16,\\
   a^* = 500 \left[f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right],\\
   b^* = 200 \left[f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right]
$$

其中$f(t)=t^{1/3}$，（若$t > 0.008856$），否则用线性近似。

2. LAB → CAM02

使用 CIE CAM02 正向模型（需输入：$ L^*, a^*, b^* $、光源 XYZ、背景/适应参数）。   输出：感知量 $ J, C, h $（明度、彩度、色相角）。

3. CAM02 → CAM02-UCS

$$
J' = \frac{1.7\,J}{1 + 0.007\,J},\quad
   a' = C \cos h,\quad
   b' = C \sin h
$$

对 **参考光源** 重复步骤 1–3，得到 $ (J'_{\text{ref},i}, a'_{\text{ref},i}, b'_{\text{ref},i}) $。对 **测试光源** 得到 $ (J'_{\text{test},i}, a'_{\text{test},i}, b'_{\text{test},i}) $。

- **计算 Rf（保真度指数）**
  对每个样本 $ i $ 计算色差：
  $$
  \Delta E_i = \sqrt{(J'_{\text{test},i} - J'_{\text{ref},i})^2 + (a'_{\text{test},i} - a'_{\text{ref},i})^2 + (b'_{\text{test},i} - b'_{\text{ref},i})^2}
  $$

平均色差：
$$
\overline{\Delta E} = \frac{1}{99} \sum_{i=1}^{99} \Delta E_i
$$

Rf 变换：

$$
Rf = 10 \ln \left[ \exp\left(\frac{100 - \overline{\Delta E}}{10}\right) + 1 \right]
$$

色域指数 $Rg$

- **计算 Rg（色域指数）**

1. **划分 16 个 hue-angle bin**：每个 bin 包含若干 CES。对每个 bin $ j = 1 \dots 16 $ 计算平均坐标：

   $$
   \bar a'_{\text{test},j} = \frac{1}{n_j} \sum_{k \in \text{bin } j} a'_{\text{test},k},\quad
   \bar b'_{\text{test},j} = \frac{1}{n_j} \sum_{k \in \text{bin } j} b'_{\text{test},k}
   $$

   （参考光源同理）
2. **多边形面积**（16 个顶点，Shoelace 公式）：

   $$
   A_{\text{test}} = \frac{1}{2} \left| \sum_{j=1}^{16} \left( \bar a'_{\text{test},j} \bar b'_{\text{test},j+1} - \bar a'_{\text{test},j+1} \bar b'_{\text{test},j} \right) \right|
   $$

   （循环：第 17 个点 = 第 1 个点）
3. **Rg 计算**：

   $$
   Rg = \frac{A_{\text{test}}}{A_{\text{ref}}} \times 100
   $$

## 生理节律效应参数

褪黑素日光照度比 $mel-DER$

可以使用官方工具箱直接输出
```python
import numpy as np
import colour

spd = ...  # ndarray 380–780 nm, 1 nm
D65 = colour.SDS_ILLUMINANTS['D65']
s_mel = colour.SDS_BASIS_FUNCTIONS_CIE_S_026['mel']

# 褪黑素辐照度
E_mel = np.trapz(spd * s_mel.values, s_mel.wavelengths)

# 明视觉照度
E_phot = np.trapz(spd * colour.SDS_PHOTOPIC_LEFS['CIE 1924'].values,
                  s_mel.wavelengths) * 683  # lx

# mel-ELR
mel_ELR = E_mel / E_phot * 1000  # mW·lm⁻¹

# mel-DER
mel_ELR_D65 = (np.trapz(D65 * s_mel.values, s_mel.wavelengths) /
               np.trapz(D65 * colour.SDS_PHOTOPIC_LEFS['CIE 1924'].values, s_mel.wavelengths)) * 1000
mel_DER = mel_ELR / mel_ELR_D65
```