# -*- coding: utf-8 -*-
"""
LTI系统算法

@author: https://github.com/zhaohaojie1998
"""
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from typing import Optional, Literal, Union, Tuple
from ..types import MatLike

# 默认数值容差，参考NumPy等数学库的常用值
DEFAULT_TOL = 1e-8

__all__ = [
    "get_controllable_matrix",
    "is_controllable",
    "get_observable_matrix",
    "is_observable",

    "is_stable",
    "solve_lyapunov",
    "is_lyapunov_stable",

    "get_uncontrollable_modes",
    "is_uncontrollable_stable",  # 可镇定性
    "get_unobservable_modes",
    "is_unobservable_stable",    # 可检测性

    "solve_algebraic_riccati",
    "solve_lqr",
    "solve_lqry",
    "solve_lqi",
]

def matrix_positive_semidefinite(mat: np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """
    判断矩阵是否半正定 (LQR允许Q半正定, R必须正定)
    半正定条件：
    1. 矩阵对称（数值容差内）
    2. 所有特征值的实部 ≥ -tol (允许极小负特征值, 容错浮点误差)
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    # 1. 判断对称性（数值容差内）
    if not np.allclose(mat, mat.T, atol=tol):
        return False
    # 2. 计算特征值，允许极小负特征值（容错浮点误差）
    eig_vals = nla.eigvals(mat)
    return all(np.real(eig) > -tol for eig in eig_vals)

def matrix_positive_definite(mat: np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """
    判断矩阵是否正定 (R矩阵必须正定)
    正定条件：
    1. 矩阵对称（数值容差内）
    2. 所有特征值的实部 > tol (严格大于0)
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    # 1. 判断对称性
    if not np.allclose(mat, mat.T, atol=tol):
        return False
    # 2. 特征值严格大于0（容差内）
    eig_vals = nla.eigvals(mat)
    return all(np.real(eig) > tol for eig in eig_vals)


# -------------------------- 线性定常系统工具函数 --------------------------

# 能控性判断
def get_controllable_matrix(A: MatLike, B: MatLike) -> MatLike:
    """
    返回系统的可控制性矩阵Sc
    
    可控制性矩阵用于判断系统是否能控. 对于n维系统, 可控制性矩阵定义为:
    Sc = [B AB A²B ... A^(n-1)B]
    
    如果rank(Sc) = n, 则系统完全能控.
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    
    Returns:
    - ndarray, 可控制性矩阵 (n x np)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    Sc = B
    for i in range(1, n):
        Sc = np.hstack((Sc, nla.matrix_power(A, i) @ B))
    return Sc

def is_controllable(A: MatLike, B: MatLike) -> bool:
    """
    判断系统是否能控
    
    系统能控性是指通过控制输入u(t), 可以在有限时间内将系统从任意初始状态转移到任意目标状态.
    
    判据: 如果可控制性矩阵的秩等于系统维数n, 则系统完全能控.
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    
    Returns:
    - bool, 系统是否能控
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Sc = get_controllable_matrix(A, B)
    rank = nla.matrix_rank(Sc)
    return rank == A.shape[0]

# 能观性判断
def get_observable_matrix(A: MatLike, C: MatLike) -> MatLike:
    """
    返回系统的可观性矩阵So
    
    可观性矩阵用于判断系统是否能观. 对于n维系统, 可观性矩阵定义为:
    So = [C; CA; CA²; ...; CA^(n-1)]
    
    如果rank(So) = n, 则系统完全能观.
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - C: ndarray, 输出矩阵 (q x n)

    Returns:
    - ndarray, 可观性矩阵 (qn x n)
    """
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    n = A.shape[0]
    So = C
    for i in range(1, n):
        So = np.vstack((So, C @ nla.matrix_power(A, i)))
    return So

def is_observable(A: MatLike, C: MatLike) -> bool:
    """
    判断系统是否能观
    
    系统可观性是指通过观测输出y(t), 可以在有限时间内确定系统的初始状态.
    
    判据: 如果可观性矩阵的秩等于系统维数n, 则系统完全能观.
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - C: ndarray, 输出矩阵 (q x n)
    
    Returns:
    - bool, 系统是否能观
    """
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    So = get_observable_matrix(A, C)
    rank = nla.matrix_rank(So)
    return rank == A.shape[0]

# 稳定性判断
def is_stable(A: MatLike, discrete: bool = False, tol: float = DEFAULT_TOL) -> bool:
    """
    判断系统是否稳定
    
    系统稳定性是指系统在受到扰动后, 能够恢复到平衡状态的能力.
    
    连续时间系统稳定条件: 所有特征值的实部小于0
    离散时间系统稳定条件: 所有特征值的模小于1
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    - tol: 数值容差, 默认DEFAULT_TOL
    
    Returns:
    - bool, 系统是否稳定
    """
    A = np.asarray(A, dtype=np.float64)
    eigenvalues = nla.eigvals(A)
    if discrete:
        # 离散系统稳定条件：所有特征值的模小于1（加入微小容差避免数值误差）
        return all(np.abs(eig) < 1 - tol for eig in eigenvalues)
    else:
        # 连续系统稳定条件：所有特征值的实部小于0
        return all(np.real(eig) < -tol for eig in eigenvalues)

def solve_lyapunov(A: MatLike, Q: MatLike, discrete: bool = False) -> MatLike:
    """
    求解Lyapunov方程

    连续时间系统Lyapunov方程: A^TP + PA = -Q , 等价于 scipy求解的 AX + XA^H = -Q
    离散时间系统Lyapunov方程: A^TPA - P = -Q , 等价于 scipy求解的 AXA^H - X = -Q
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - Q: ndarray, 正定矩阵 (n x n)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    
    Returns:
    - ndarray, Lyapunov方程的解P (n x n)
    """
    A = np.asarray(A, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    assert A.shape == Q.shape, f"A和Q必须是相同形状的矩阵, 当前A形状: {A.shape}, Q形状: {Q.shape}"
    
    if discrete:
        # scipy离散Lyapunov求解的是 AXA^H - X + Q = 0 → 对应 A^TPA - P = -Q
        P = sla.solve_discrete_lyapunov(A, Q)
    else:
        # scipy连续Lyapunov求解的是 AX + XA^H = Q → 需传入-Q以匹配 A^TP + PA = -Q
        P = sla.solve_continuous_lyapunov(A, -Q)
    return P

def is_lyapunov_stable(A: MatLike, Q: MatLike, discrete: bool = False, tol: float = DEFAULT_TOL) -> bool:
    """
    判断系统是否Lyapunov稳定
    
    如果存在正定矩阵P满足Lyapunov方程, 则系统Lyapunov稳定.

    连续时间系统Lyapunov方程: A^TP + PA = -Q
    离散时间系统Lyapunov方程: A^TPA - P = -Q
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - Q: ndarray, 正定矩阵 (n x n)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    - tol: 数值容差, 默认DEFAULT_TOL
    
    Returns:
    - bool, 系统是否Lyapunov稳定
    """
    try:
        # 求解Lyapunov方程
        P = solve_lyapunov(A, Q, discrete)
        # 检查P是否正定
        # 只需要判断特征值是否大于0，不需要判断对称，因为Lyapunov方程解是对称的
        eig_vals_P = nla.eigvals(P)
        return all(np.real(eig) > tol for eig in eig_vals_P)
    except Exception as e:
        print(f"Lyapunov稳定性判断出错: {e}")
        return False
    
# 不可控模态与可镇定性
def get_uncontrollable_modes(A: MatLike, B: MatLike) -> MatLike:
    """
    求解LTI系统的不可控模态 (基于能控性矩阵的零空间)
    
    原理：
    1. 计算能控性矩阵Sc的秩, 得到能控子空间维数
    2. 对Sc^T进行SVD分解, 得到能控性矩阵的零空间 (不可控子空间)
    3. 将状态矩阵A投影到不可控子空间, 其特征值即为不可控模态
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    
    Returns:
    - ndarray, 不可控模态 (特征值) , 若系统完全能控则返回空数组
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    # 步骤1：计算能控性矩阵
    Sc = get_controllable_matrix(A, B)
    rank_Sc = nla.matrix_rank(Sc)
    n = A.shape[0]

    # 若完全能控，无不可控模态
    if rank_Sc == n:
        return np.array([], dtype=np.float64)
    
    # 步骤2：求解Sc^T的零空间（不可控子空间）
    # SVD分解：Sc^T = UΣV^T，零空间对应V中Σ=0的列
    U, S, Vh = nla.svd(Sc.T, full_matrices=True)
    # 零空间基（V的后n-rank列）
    uncontrollable_basis = Vh[rank_Sc:].T
    
    # 步骤3：将A投影到不可控子空间，计算其特征值（不可控模态）
    # 投影矩阵：A_un = T^T A T，其中T是不可控子空间基
    A_un = uncontrollable_basis.T @ A @ uncontrollable_basis
    uncontrollable_modes = nla.eigvals(A_un)
    
    # 去重（避免数值误差导致的重复模态）
    uncontrollable_modes = np.unique(np.round(uncontrollable_modes, 10))
    
    return uncontrollable_modes

def is_uncontrollable_stable(A: MatLike, B: MatLike, discrete: bool = False, tol: float = DEFAULT_TOL) -> bool:
    """
    判断LTI系统的镇定性 (不可控模态是否稳定)
    
    可镇定性定义：系统的所有不可控模态都是稳定的，则系统可镇定
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    - tol: 数值容差, 默认DEFAULT_TOL
    
    Returns:
    - bool: 
      - 完全能控 → True
      - 存在不可控模态 → 所有不可控模态稳定则返回True, 否则False
    """
    uncontrollable_modes = get_uncontrollable_modes(A, B)
    
    # 完全能控，无不可控模态
    if len(uncontrollable_modes) == 0:
        return True
    
    # 判断不可控模态稳定性（加入容差避免数值误差）
    if discrete:
        # 离散系统：所有不可控模态的模<1
        return all(np.abs(mode) < 1 - tol for mode in uncontrollable_modes)
    else:
        # 连续系统：所有不可控模态的实部<0
        return all(np.real(mode) < -tol for mode in uncontrollable_modes)

# 不可观模态与可检测性
def get_unobservable_modes(A: MatLike, C: MatLike) -> MatLike:
    """
    求解LTI系统的不可观模态 (基于能观性矩阵的零空间)
    
    原理：
    1. 计算能观性矩阵So的秩, 得到能观子空间维数
    2. 对So进行SVD分解, 得到能观性矩阵的零空间 (不可观子空间)
    3. 将状态矩阵A投影到不可观子空间, 其特征值即为不可观模态
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - C: ndarray, 输出矩阵 (q x n)
    
    Returns:
    - ndarray, 不可观模态 (特征值) , 若系统完全能观则返回空数组
    """
    A = np.array(A, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    # 步骤1：计算能观性矩阵
    So = get_observable_matrix(A, C)
    rank_So = nla.matrix_rank(So)
    n = A.shape[0]

    # 若完全能观，无不可观模态
    if rank_So == n:
        return np.array([], dtype=np.float64)
    
    # 步骤2：求解So的零空间（不可观子空间）
    # SVD分解：So = UΣV^T，零空间对应V中Σ=0的列
    U, S, Vh = nla.svd(So, full_matrices=True)
    # 零空间基（V的后n-rank列）
    unobservable_basis = Vh[rank_So:].T
    
    # 步骤3：将A投影到不可观子空间，计算其特征值（不可观模态）
    # 投影矩阵：A_un = T^T A T，其中T是不可观子空间基
    A_un = unobservable_basis.T @ A @ unobservable_basis
    unobservable_modes = nla.eigvals(A_un)
    
    # 去重（避免数值误差导致的重复模态）
    unobservable_modes = np.unique(np.round(unobservable_modes, 10))
    
    return unobservable_modes

def is_unobservable_stable(A: MatLike, C: MatLike, discrete: bool = False, tol: float = DEFAULT_TOL) -> bool:
    """
    判断LTI系统的可检测性 (不可观模态是否稳定)
    
    可检测性定义：系统的所有不可观模态都是稳定的，则系统可检测
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - C: ndarray, 输出矩阵 (q x n)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    - tol: 数值容差, 默认DEFAULT_TOL
    
    Returns:
    - bool: 
      - 完全能观 → True
      - 存在不可观模态 → 所有不可观模态稳定则返回True, 否则False
    """
    unobservable_modes = get_unobservable_modes(A, C)
    
    # 完全能观，无不可观模态
    if len(unobservable_modes) == 0:
        return True
    
    # 判断不可观模态稳定性（加入容差避免数值误差）
    if discrete:
        # 离散系统：所有不可观模态的模<1
        return all(np.abs(mode) < 1 - tol for mode in unobservable_modes)
    else:
        # 连续系统：所有不可观模态的实部<0
        return all(np.real(mode) < -tol for mode in unobservable_modes)

# LQR问题求解
def solve_algebraic_riccati(A: MatLike, B: MatLike, Q: MatLike, R: MatLike, discrete: bool = False) -> MatLike:
    """
    求解代数黎卡提方程
    
    连续时间代数黎卡提方程：
    A^TP + PA - (PB) R^(-1) (B^TP) + Q = 0
    
    离散时间代数黎卡提方程：
    A^TPA - P - (A^TPB) (R + B^TPB)^(-1) (B^TPA) + Q = 0
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    - Q: ndarray, 状态权重矩阵 (n x n)
    - R: ndarray, 控制输入权重矩阵 (p x p)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    
    Returns:
    - ndarray, Riccati方程的解P (n x n)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    
    try:
        if discrete:
            P = sla.solve_discrete_are(A, B, Q, R)
        else:
            P = sla.solve_continuous_are(A, B, Q, R)
        return P
    
    except nla.LinAlgError as e:
        stabilizable = is_uncontrollable_stable(A, B, discrete=discrete)
        is_Q_psd = matrix_positive_semidefinite(Q)
        is_R_pd = matrix_positive_definite(R)

        error_msg = []
        if not stabilizable:
            error_msg.append("系统不可镇定性（存在不稳定的不可控模态）")
        if not is_Q_psd:
            error_msg.append("状态权重矩阵Q非半正定")
        if not is_R_pd:
            error_msg.append("控制输入权重矩阵R非正定")
        
        if error_msg:
            raise nla.LinAlgError(f"代数黎卡提方程求解失败：{'; '.join(error_msg)}") from e
        else:
            raise nla.LinAlgError(f"代数黎卡提方程求解失败：{str(e)}（未知原因）") from e
        
def solve_lqr(
    A: MatLike, B: MatLike,
    Q: MatLike, R: MatLike, discrete: bool = False
) -> Tuple[MatLike, MatLike, MatLike, bool]:
    """
    无限时域LQR状态调节问题求解器, 返回[K, P, λ, stable]
    
    连续时间: 
    - 性能指标: J = ∫(x'Qx + u'Ru)dt
    - 控制律: u = -Kx
    - 增益K: K = R^(-1) * B^T * P
    
    离散时间:
    - 性能指标: J = Σ(x'Qx + u'Ru)
    - 控制律: u = -Kx
    - 增益K: K = (R + B^T P B)^(-1) * B^T * P * A

    系统模型:
    - dx/dt = Ax + Bu

    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    - Q: ndarray, 状态权重矩阵 (n x n)
    - R: ndarray, 控制输入权重矩阵 (p x p)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    
    Returns:
    - K: ndarray, 状态反馈增益矩阵 (p x n)
    - P: ndarray, Riccati方程的解 (n x n)
    - λ: ndarray, 闭环系统特征值 (n,)
    - stable: bool, 闭环系统是否稳定
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    
    P = solve_algebraic_riccati(A, B, Q, R, discrete)
    if discrete:
        # 离散系统LQR增益
        K = nla.inv(R + B.T @ P @ B) @ B.T @ P @ A
        # 闭环系统矩阵
        A_cl = A - B @ K
        λ = nla.eigvals(A_cl)
        # 离散系统稳定性判断
        stable = all(np.abs(mode) < 1 - DEFAULT_TOL for mode in λ)
    else:
        # 连续系统LQR增益
        K = nla.inv(R) @ B.T @ P
        # 闭环系统矩阵
        A_cl = A - B @ K
        λ = nla.eigvals(A_cl)
        # 连续系统稳定性判断
        stable = all(np.real(mode) < -DEFAULT_TOL for mode in λ)
    return K, P, λ, stable

def solve_lqry(
    A: MatLike, B: MatLike, C: MatLike,
    Qy: MatLike, R: MatLike, discrete: bool = False
) -> Tuple[MatLike, MatLike, MatLike, bool]:
    """
    无限时域LQR输出调节问题求解器, 返回[K, P, λ, stable], 不支持带D的系统模型
    
    连续时间:
    - 性能指标: J = ∫(y'Qy + u'Ru)dt
    - 控制律: u = -Kx
    - 增益K: K = R^(-1) * B^T * P
    
    离散时间:
    - 性能指标: J = Σ(y'Qy + u'Ru)
    - 控制律: u = -Kx
    - 增益K: K = (R + B^T P B)^(-1) * B^T * P * A

    系统模型:
    - dx/dt = Ax + Bu
    - y = Cx

    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    - C: ndarray, 输出矩阵 (q x n)
    - Qy: ndarray, 输出权重矩阵 (q x q)
    - R: ndarray, 控制输入权重矩阵 (p x p)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    
    Returns:
    - K: ndarray, 状态反馈增益矩阵 (p x n)
    - P: ndarray, Riccati方程的解 (n x n)
    - λ: ndarray, 闭环系统特征值 (n,)
    - stable: bool, 闭环系统是否稳定
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Qy = np.asarray(Qy, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    
    Q = C.T @ Qy @ C  # 等效状态调节器权重矩阵
    return solve_lqr(A, B, Q, R, discrete)

def solve_lqi(
    A: MatLike, B: MatLike, C: MatLike, D: MatLike,
    Qy: MatLike, R: MatLike, discrete: bool = False,
    handle_cross_terms: bool = False
) -> Tuple[MatLike, MatLike, MatLike, MatLike, bool]:
    """
    无限时域LQR输出跟踪问题求解器 (积分型), 返回[Kx, Ki, P, λ, stable]

    连续时间:
    - 性能指标: J = ∫[(r-y)'Qy(r-y) + u'Ru)]dt
    - 控制律: u = -Kx * x - Ki * ∫(r-y)dt

    离散时间:
    - 性能指标: J = Σ(r-y)Qy(r-y) + u'Ru)
    - 控制律: u = -Kx * x - Ki * Σ(r-y)

    系统模型:
    - dx/dt = Ax + Bu
    - y = Cx + Du

    引入误差积分 i = ∫(r-y)dt:
    - di/dt = r - y = -Cx - Du + r

    构建增广状态 z = [[x], [i]]
        dx/dt = Ax + Bu
        di/dt = -Cx - Du + r (先假设r=0)

    扩展状态矩阵:
        [A   0]
        [-C  0]
    扩展输入矩阵:
        [B]
        [-D]
    扩展权重矩阵Q_z:
        [0   0]
        [0  Qy]
    
    Parameters:
    - A: ndarray, 状态矩阵 (n x n)
    - B: ndarray, 输入矩阵 (n x p)
    - C: ndarray, 输出矩阵 (q x n)
    - D: ndarray, 直接传递矩阵 (q x p)
    - Qy: ndarray, 输出权重矩阵 (q x q)
    - R: ndarray, 控制输入权重矩阵 (p x p)
    - discrete: bool, 是否为离散时间系统 (默认为False)
    - handle_cross_terms: bool, 是否考虑x和u的交叉项 2 x^T C^T Qy D u (默认为False)

    Returns:
    - Kx: ndarray, 状态反馈增益矩阵 (p x n)
    - Ki: ndarray, 误差积分增益矩阵 (p x q)
    - P: ndarray, Riccati方程的解 ((n+q) x (n+q))
    - λ: ndarray, 闭环系统特征值 (n+q,)
    - stable: bool, 闭环系统是否稳定
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    Qy = np.asarray(Qy, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    
    n = A.shape[0]  # 原状态维度
    q = C.shape[0]  # 输出维度
    p = B.shape[1]  # 输入维度
    
    # 1. 构造扩展状态空间
    # 扩展状态矩阵 A_z = [[A, 0], [-C, 0]]
    A_z = np.block([
        [A, np.zeros((n, q))],
        [-C, np.zeros((q, q))]
    ])
    
    # 扩展输入矩阵 B_z = [[B], [-D]]
    B_z = np.block([
        [B],
        [-D]
    ])
    
    # 2. 构造扩展权重矩阵
    # 扩展权重矩阵Q_z = [[0, 0], [0, Qy]], 适合无静差跟踪。 另一种实现 Q_z = [[C^TQyC, 0], [0, 0]], 适合输出调节问题
    Q_z = np.block([
        [np.zeros((n, n)), np.zeros((n, q))],
        [np.zeros((q, n)), Qy]
    ])
    
    # 3. 构造扩展控制权重矩阵R_z
    R_z = R + D.T @ Qy @ D

    # 4.由D矩阵产生的交叉项: 2 x^T N u, 其中 N = C^T Qy D
    N = C.T @ Qy @ D # (n, p)
    
    # 处理由D产生的交叉项
    if handle_cross_terms:
        N_z = np.zeros((n + q, p)) # (n+q, p)
        N_z[:n, :] = N # 后q行为0
        try:
            # 计算变换矩阵 R_inv_NT = R_z^{-1} N^T
            R_inv = np.linalg.inv(R_z)
            R_inv_NT = R_inv @ N_z.T
            
            # 构造变换后的系统矩阵
            A_transformed = A_z - B_z @ R_inv_NT
            B_transformed = B_z
            Q_transformed = Q_z - N_z @ R_inv @ N_z.T
            
            # 求解变换后的Riccati方程
            K_transformed, P, λ, stable = solve_lqr(A_transformed, B_transformed, Q_transformed, R_z, discrete)
            
            # 恢复原控制律 K_z = K_transformed + R_inv_NT
            K_z = K_transformed + R_inv_NT
        
        except np.linalg.LinAlgError as e:
            print(f"[warning] 交叉项处理失败: {e}, 不考虑交叉项再次求解")
            K_z, P, λ, stable = solve_lqr(A_z, B_z, Q_z, R_z, discrete)
    
    # 不处理交叉项，直接使用solve_lqr
    else:
        K_z, P, λ, stable = solve_lqr(A_z, B_z, Q_z, R_z, discrete)
    
    # 5. 分解增益矩阵 K_z = [Kx, Ki]
    Kx = K_z[:, :n]
    Ki = K_z[:, n:]
    
    return Kx, Ki, P, λ, stable




# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 测试连续系统
    print("=== 连续系统测试 ===")
    A_cont = np.array([[0, 1], [-2, -3]])
    B_cont = np.array([[0], [1]])
    C_cont = np.array([[1, 0]])
    Q_cont = np.eye(2)
    R_cont = np.array([[1]])
    Qy_cont = np.array([[1]])
    
    # 能控性测试
    print(f"能控性: {is_controllable(A_cont, B_cont)}")
    print(f"不可控模态: {get_uncontrollable_modes(A_cont, B_cont)}")
    print(f"可镇定性: {is_uncontrollable_stable(A_cont, B_cont)}")
    
    # 能观性测试
    print(f"能观性: {is_observable(A_cont, C_cont)}")
    print(f"不可观模态: {get_unobservable_modes(A_cont, C_cont)}")
    print(f"可检测性: {is_unobservable_stable(A_cont, C_cont)}")
    
    # LQR测试
    K, P, λ, stable = solve_lqr(A_cont, B_cont, Q_cont, R_cont)
    print(f"LQR增益K: \n{K}")
    print(f"闭环稳定性: {stable}")

    # LQRy
    K, P, λ, stable = solve_lqry(A_cont, B_cont, C_cont, Qy_cont, R_cont)
    print(f"LQRy增益K: \n{K}")
    print(f"闭环稳定性: {stable}")
    
    # LQI测试
    D_cont = np.array([[0.1]])  # 直接传递矩阵，单输入单输出系统
    Kx, Ki, P_i, λ_i, stable_i = solve_lqi(A_cont, B_cont, C_cont, D_cont, Qy_cont, R_cont)
    print(f"LQI状态增益Kx: \n{Kx}")
    print(f"LQI积分增益Ki: \n{Ki}")
    print(f"闭环稳定性: {stable_i}")
    
    # 测试考虑交叉项的情况
    Kx_cross, Ki_cross, P_i_cross, λ_i_cross, stable_i_cross = solve_lqi(A_cont, B_cont, C_cont, D_cont, Qy_cont, R_cont, handle_cross_terms=True)
    print(f"考虑交叉项的LQI状态增益Kx: \n{Kx_cross}")
    print(f"考虑交叉项的LQI积分增益Ki: \n{Ki_cross}")
    print(f"闭环稳定性: {stable_i_cross}")
    
    # 测试离散系统
    print("\n=== 离散系统测试 ===")
    A_disc = np.array([[0.5, 0], [0, 1.2]])
    B_disc = np.array([[1], [0]])
    C_disc = np.array([[1, 0]])
    Q_disc = np.eye(2)
    R_disc = np.array([[1]])
    Qy_disc = np.array([[1]])
    
    # 能控性测试
    print(f"能控性: {is_controllable(A_disc, B_disc)}")
    print(f"不可控模态: {get_uncontrollable_modes(A_disc, B_disc)}")
    print(f"可镇定性: {is_uncontrollable_stable(A_disc, B_disc, discrete=True)}")

    # 能观性测试
    print(f"能观性: {is_observable(A_disc, C_disc)}")
    print(f"不可观模态: {get_unobservable_modes(A_disc, C_disc)}")
    print(f"可检测性: {is_unobservable_stable(A_disc, C_disc, discrete=True)}")

    # LQR测试
    K_d, P_d, λ_d, stable_d = solve_lqr(A_disc, B_disc, Q_disc, R_disc, discrete=True)
    print(f"LQR增益K_d: \n{K_d}")
    print(f"闭环稳定性: {stable_d}")

    # LQRy测试
    K_d, P_d, λ_d, stable_d = solve_lqry(A_disc, B_disc, C_disc, Qy_disc, R_disc, discrete=True)
    print(f"LQRy增益K_d: \n{K_d}")
    print(f"闭环稳定性: {stable_d}")
    
    # LQI测试
    D_disc = np.array([[0.1]])  # 直接传递矩阵，单输入单输出系统
    Kx_d, Ki_d, P_i_d, λ_i_d, stable_i_d = solve_lqi(A_disc, B_disc, C_disc, D_disc, Qy_disc, R_disc, discrete=True)
    print(f"LQI状态增益Kx_d: \n{Kx_d}")
    print(f"LQI积分增益Ki_d: \n{Ki_d}")
    print(f"闭环稳定性: {stable_i_d}")
    
    # 测试考虑交叉项的情况
    Kx_d_cross, Ki_d_cross, P_i_d_cross, λ_i_d_cross, stable_i_d_cross = solve_lqi(A_disc, B_disc, C_disc, D_disc, Qy_disc, R_disc, discrete=True, handle_cross_terms=True)
    print(f"考虑交叉项的LQI状态增益Kx_d: \n{Kx_d_cross}")
    print(f"考虑交叉项的LQI积分增益Ki_d: \n{Ki_d_cross}")
    print(f"闭环稳定性: {stable_i_d_cross}")
