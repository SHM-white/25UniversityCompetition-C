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

## 颜色还原参数

保真度指数 $Rf$

色域指数 $Rg$

## 生理节律效应参数

褪黑素日光照度比 $mel-DER$
