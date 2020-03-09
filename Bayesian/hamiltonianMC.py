
# %%
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
mu = 1
sigma=1

# %% Log probability
# ここでは平均mu, 標準偏差sigmaの正規分布
# -1をかけると物理でいうポテンシャルエネルギー
def log_normal(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - (x-mu)**2/(2*sigma**2)

# Log probabilityの導関数
def d_log_normal(x, mu, sigma):
    return -(x-mu)/sigma**2

# 運動エネルギー
def momentum(p, tau):
    return p**2/(2*tau**2)

# 運動エネルギーの導関数
def d_momentum(p, tau):
    return  p/tau**2

# ハミルトニアン
# 運動エネルギーとポテンシャルエネルギーの和
def Hamiltonian(x, p, tau):
    global mu, sigma

    return momentum(p, tau) + (-1.*log_normal(x, mu, sigma))

# リープ・フロッグ法を事項する関数
def proceed_leapflog(epsilon, x, p, tau):
    global mu, sigma

    x += -0.5*epsilon*(-1.*d_momentum(p, tau))
    p += epsilon*d_log_normal(x, mu, sigma)
    x += -epsilon*(-1.*d_momentum(p, tau))

    return x, p

# HMCを１ステップ実行するサブルーチン
def proceed_HMC_iteration(x, tau, epsilon, T):
    p = np.random.normal(0, tau, size=1)[0]
    p_new = p
    x_new= x
    for t in range(T):
        x_new, p_new = proceed_leapflog(epsilon, x_new, p_new, tau)

    alpha = np.exp(Hamiltonian(x, p, tau) - Hamiltonian(x_new, p_new, tau))
    u = np.random.uniform()
    if u < alpha:
        x_accepted = x_new
    else:
        x_accepted = x

    return x_accepted

# HMCを実行する関数
def proceed_HMC(tau, epsilon, T, ite, init):
    # Initialize
    x = [init]
    for i in range(ite):
        x.append(proceed_HMC_iteration(x[i], tau, epsilon, T))

    return x

init = 100
theta = proceed_HMC(tau=2, epsilon=0.1, T=10, ite=2000, init=init)

plt.plot(theta)
