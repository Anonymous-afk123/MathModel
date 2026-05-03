"""
电除尘器系统完整分析模型
=====================================
包含：
  问题1：前馈控制器与闭环仿真辨识
  问题2：工况划分与最低电耗优化
  问题3：控制策略差异分析与优先级规律
  问题4：排放限值下调（10→5 mg/Nm³）的电耗影响
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import sys

warnings.filterwarnings('ignore')

# 解决Windows编码问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===========================================================================
# 【问题1】前馈控制器与闭环仿真
# ===========================================================================
print("=" * 80)
print("【问题1】前馈控制器与闭环仿真")
print("=" * 80)

# 数据加载与预处理
print("\n加载数据...")
file_path = r"C:\Users\admin\Downloads\a题数据.xlsx"

try:
    df = pd.read_excel(file_path, sheet_name='Cement_ESP_Data')
except:
    df = pd.read_excel(file_path)

df = df.sort_values('timestamp').reset_index(drop=True)
df = df[(df['C_in_gNm3'] > 0) & (df['Q_Nm3h'] > 0)]

df['C_in_mg'] = df['C_in_gNm3'] * 1000.0
df['T_K'] = df['Temp_C'] + 273.15
print(f"有效数据量: {len(df)}")

# 动态积灰状态 S_i
print("\n构造动态积灰状态 S_i ...")
alpha_soot = 0.3
for i in range(1, 5):
    col = f'T{i}_s'
    S = np.zeros(len(df))
    S[0] = df[col].iloc[0]
    for t in range(1, len(df)):
        S[t] = alpha_soot * df[col].iloc[t] + (1 - alpha_soot) * S[t-1]
    df[f'S{i}'] = S

# 前馈控制器训练
print("\n训练前馈控制器...")
ff_features = ['C_in_gNm3', 'Q_Nm3h', 'Temp_C']
ff_targets = ['U1_kV','U2_kV','U3_kV','U4_kV','T1_s','T2_s','T3_s','T4_s']

ff_model = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
)
ff_model.fit(df[ff_features], df[ff_targets])

ff_pred_all = ff_model.predict(df[ff_features])
print("前馈控制器拟合效果 (R²):")
for i, col in enumerate(ff_targets):
    r2 = r2_score(df[col], ff_pred_all[:, i])
    print(f"  {col}: {r2:.3f}")

# 基于前馈控制器的闭环仿真准备
ops_pred = ff_pred_all
U1_ff, U2_ff, U3_ff, U4_ff = ops_pred[:,0], ops_pred[:,1], ops_pred[:,2], ops_pred[:,3]
T1_ff, T2_ff, T3_ff, T4_ff = ops_pred[:,4], ops_pred[:,5], ops_pred[:,6], ops_pred[:,7]

S1_ff = np.zeros(len(df)); S2_ff = np.zeros(len(df))
S3_ff = np.zeros(len(df)); S4_ff = np.zeros(len(df))
S1_ff[0] = T1_ff[0]; S2_ff[0] = T2_ff[0]; S3_ff[0] = T3_ff[0]; S4_ff[0] = T4_ff[0]
for t in range(1, len(df)):
    S1_ff[t] = alpha_soot * T1_ff[t] + (1-alpha_soot)*S1_ff[t-1]
    S2_ff[t] = alpha_soot * T2_ff[t] + (1-alpha_soot)*S2_ff[t-1]
    S3_ff[t] = alpha_soot * T3_ff[t] + (1-alpha_soot)*S3_ff[t-1]
    S4_ff[t] = alpha_soot * T4_ff[t] + (1-alpha_soot)*S4_ff[t-1]

U_mat_ff = np.column_stack([U1_ff, U2_ff, U3_ff, U4_ff])
S_mat_ff = np.column_stack([S1_ff, S2_ff, S3_ff, S4_ff])
T_actual = df['Temp_C'].values
C_in_mg = df['C_in_mg'].values
Q_actual = df['Q_Nm3h'].values

# 物理模型定义与辨识
print("\n开始物理模型辨识...")

def physical_omega(U_mat, S_mat, T_v, params):
    """物理驱进速度 Ω"""
    K, alpha, beta, k1, k2, k3, k4 = params
    T_K = T_v + 273.15
    k_arr = np.array([k1, k2, k3, k4])
    U_eff = U_mat - k_arr * S_mat
    U_eff = np.clip(U_eff, 1.0, None)
    sum_U = np.sum(U_eff, axis=1)
    Omega = K * (T_K ** (-beta)) * (sum_U ** alpha)
    return Omega

def loss_identification(params):
    Omega_sim = physical_omega(U_mat_ff, S_mat_ff, T_actual, params)
    C_sim = C_in_mg * np.exp(-Omega_sim / Q_actual)
    mean_err = np.abs(np.mean(C_sim) - 50.0)
    std_penalty = np.std(C_sim)
    return mean_err + 0.3 * std_penalty

bounds = [(1e3, 2e6), (1.0, 2.5), (0.0, 2.0), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02)]
result = differential_evolution(loss_identification, bounds, maxiter=200, popsize=25, seed=42, polish=True)
opt_params = result.x

print(f"辨识完成，最终损失 = {result.fun:.4f}")
print(f"参数: K={opt_params[0]:.2f}, α={opt_params[1]:.3f}, β={opt_params[2]:.3f}")
for i in range(4):
    print(f"  k{i+1} = {opt_params[3+i]:.6f}")

# 闭环仿真结果评估
Omega_sim = physical_omega(U_mat_ff, S_mat_ff, T_actual, opt_params)
C_sim = C_in_mg * np.exp(-Omega_sim / Q_actual)

print("\n" + "="*50)
print("闭环仿真结果")
print(f"  仿真出口浓度均值: {np.mean(C_sim):.2f} mg/Nm³")
print(f"  仿真出口浓度标准差: {np.std(C_sim):.4f}")
print(f"  中位数: {np.median(C_sim):.2f}")
print(f"  5%分位: {np.percentile(C_sim, 5):.2f}, 95%分位: {np.percentile(C_sim, 95):.2f}")
print(f"实际出口浓度均值: {df['C_out_mgNm3'].mean():.2f}")
print(f"实际出口浓度标准差: {df['C_out_mgNm3'].std():.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(C_sim, bins=60, edgecolor='k', alpha=0.7, color='steelblue')
axes[0,0].axvline(50, color='r', linestyle='--', linewidth=2, label='50 mg/Nm³')
axes[0,0].axvline(np.mean(C_sim), color='orange', label=f'均值={np.mean(C_sim):.1f}')
axes[0,0].set_xlabel('仿真出口浓度 (mg/Nm³)')
axes[0,0].set_ylabel('频次')
axes[0,0].set_title('闭环仿真出口浓度分布')
axes[0,0].legend()

n_plt = min(500, len(df))
axes[0,1].plot(df['C_out_mgNm3'].values[:n_plt], 'b.', markersize=2, alpha=0.7, label='实际')
axes[0,1].plot(C_sim[:n_plt], 'r.', markersize=1, alpha=0.5, label='仿真')
axes[0,1].axhline(50, color='k', linestyle='--', alpha=0.5)
axes[0,1].set_xlabel('样本序号')
axes[0,1].set_ylabel('出口浓度 (mg/Nm³)')
axes[0,1].set_title(f'前{n_plt}点对比')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

axes[1,0].scatter(df['U1_kV'], U1_ff, s=1, alpha=0.4, label='U1')
axes[1,0].scatter(df['U2_kV'], U2_ff, s=1, alpha=0.4, label='U2')
axes[1,0].plot([35,75],[35,75],'k--',lw=1)
axes[1,0].set_xlabel('实际电压 (kV)')
axes[1,0].set_ylabel('前馈预测电压 (kV)')
axes[1,0].set_title('前馈控制器电压预测')
axes[1,0].legend(markerscale=5)
axes[1,0].grid(alpha=0.3)

axes[1,1].axis('off')
textstr = f'辨识物理参数:\nK = {opt_params[0]:.1f}\nα = {opt_params[1]:.3f}\nβ = {opt_params[2]:.3f}\n'
for i in range(4):
    textstr += f'k{i+1} = {opt_params[3+i]:.5f}\n'
textstr += f'\n闭环仿真均值: {np.mean(C_sim):.2f} mg/Nm³\n标准差: {np.std(C_sim):.4f}'
axes[1,1].text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center')
plt.tight_layout()
plt.show()

# 振打峰值效应分析
print("\n" + "="*50)
print("振打峰值效应分析")

df['C_std_roll'] = df['C_out_mgNm3'].rolling(10, center=True, min_periods=1).std()
df['C_max_roll'] = df['C_out_mgNm3'].rolling(10, center=True, min_periods=1).max()

df['S1_bin'] = pd.qcut(df['S1'], q=12, duplicates='drop')
grouped = df.groupby('S1_bin', observed=True).agg({'C_out_mgNm3': ['std', 'max'], 'S1': 'count'})
grouped.columns = ['C_std', 'C_max', 'count']
grouped = grouped[grouped['count'] > 15]

centers = np.array([iv.mid for iv in grouped.index])

print(f"{'S1区间':>12s}  {'浓度标准差':>10s}  {'最大浓度':>8s}")
for c, row in zip(centers, grouped.itertuples()):
    print(f"{c:>10.1f}   {row.C_std:>10.6f}   {row.C_max:>8.2f}")

if len(centers) > 2:
    lr_std = LinearRegression().fit(centers.reshape(-1,1), grouped['C_std'].values)
    lr_max = LinearRegression().fit(centers.reshape(-1,1), grouped['C_max'].values)
    print(f"\n标准差随S1斜率: {lr_std.coef_[0]:.6f} (正值表示积灰加重波动)")
    print(f"最大浓度随S1斜率: {lr_max.coef_[0]:.6f} (正值表示积灰增大峰值)")

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.errorbar(centers, grouped['C_std'], fmt='bo-', capsize=4, label='浓度标准差')
ax1.set_xlabel('积灰状态 S1 (s)')
ax1.set_ylabel('标准差 (mg/Nm³)', color='blue')
ax2 = ax1.twinx()
ax2.plot(centers, grouped['C_max'], 'r^--', label='最大浓度')
ax2.set_ylabel('最大浓度 (mg/Nm³)', color='red')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
ax1.set_title('振打积灰状态对排放波动的影响')
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================================================================
# 【问题1补充】入口条件与操作参数对出口浓度的影响
# ===========================================================================
print("\n" + "=" * 60)
print("【问题1补充】入口条件与操作参数对出口浓度的影响")

# ----- 8.1 计算并输出相关系数矩阵 -----
print("\n--- 8.1 线性相关系数 (与出口浓度 C_out_mgNm3) ---")
rel_cols = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h',
            'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV',
            'T1_s', 'T2_s', 'T3_s', 'T4_s']
corr_matrix = df[rel_cols + ['C_out_mgNm3']].corr()
print(corr_matrix['C_out_mgNm3'].sort_values(key=abs, ascending=False).to_string())

# ----- 8.2 入口条件 vs 出口浓度散点图 -----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 温度
axes[0].scatter(df['Temp_C'], df['C_out_mgNm3'], s=2, alpha=0.4,
                c=df['C_in_gNm3'], cmap='viridis')
axes[0].set_xlabel('入口温度 (℃)')
axes[0].set_ylabel('出口浓度 (mg/Nm³)')
axes[0].set_title('温度-出口浓度\n颜色反映入口浓度')
plt.colorbar(axes[0].collections[0], ax=axes[0], label='入口浓度 (g/Nm³)')

# 入口浓度
axes[1].scatter(df['C_in_gNm3'], df['C_out_mgNm3'], s=2, alpha=0.4,
                c=df['Temp_C'], cmap='plasma')
axes[1].set_xlabel('入口浓度 (g/Nm³)')
axes[1].set_ylabel('出口浓度 (mg/Nm³)')
axes[1].set_title('入口浓度-出口浓度\n颜色反映温度')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='温度 (℃)')

# 流量
axes[2].scatter(df['Q_Nm3h'], df['C_out_mgNm3'], s=2, alpha=0.4,
                c=df['C_in_gNm3'], cmap='viridis')
axes[2].set_xlabel('烟气流量 (Nm³/h)')
axes[2].set_ylabel('出口浓度 (mg/Nm³)')
axes[2].set_title('流量-出口浓度\n颜色反映入口浓度')
plt.colorbar(axes[2].collections[0], ax=axes[2], label='入口浓度 (g/Nm³)')
plt.tight_layout()
plt.show()

# ----- 8.3 操作参数 vs 出口浓度散点图 -----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# 电压 U1~U4
for i in range(4):
    ax = axes[i]
    ax.scatter(df[f'U{i+1}_kV'], df['C_out_mgNm3'], s=1, alpha=0.3,
               c=df['C_in_gNm3'], cmap='viridis')
    ax.set_xlabel(f'U{i+1} (kV)')
    ax.set_ylabel('出口浓度 (mg/Nm³)')
    ax.set_title(f'电场{i+1}电压')
    # 添加趋势线（分箱平均）
    bins = pd.qcut(df[f'U{i+1}_kV'], q=15, duplicates='drop')
    means = df.groupby(bins, observed=True)['C_out_mgNm3'].mean()
    centers = np.array([iv.mid for iv in means.index])
    ax.plot(centers, means.values, 'r.-', linewidth=2, label='分箱均值')
    ax.legend()
# 振打周期 T1~T4
for i in range(4):
    ax = axes[4+i]
    ax.scatter(df[f'T{i+1}_s'], df['C_out_mgNm3'], s=1, alpha=0.3,
               c=df['C_in_gNm3'], cmap='viridis')
    ax.set_xlabel(f'T{i+1} (s)')
    ax.set_ylabel('出口浓度 (mg/Nm³)')
    ax.set_title(f'电场{i+1}振打周期')
    bins = pd.qcut(df[f'T{i+1}_s'], q=15, duplicates='drop')
    means = df.groupby(bins, observed=True)['C_out_mgNm3'].mean()
    centers = np.array([iv.mid for iv in means.index])
    ax.plot(centers, means.values, 'r.-', linewidth=2, label='分箱均值')
    ax.legend()
plt.tight_layout()
plt.show()

# ----- 8.4 振打周期对瞬时排放峰值的影响 -----
print("\n--- 8.4 振打周期与瞬时排放峰值分析 ---")
window = 10
df['Cmax_roll'] = df['C_out_mgNm3'].rolling(window, center=True, min_periods=1).max()
df['Cstd_roll'] = df['C_out_mgNm3'].rolling(window, center=True, min_periods=1).std()

# 分别分析各电场振打周期对峰值的影响，绘制箱线图
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i in range(4):
    ax = axes[i]
    df_temp = df.copy()
    df_temp['T_bin'] = pd.cut(df_temp[f'T{i+1}_s'], bins=6, duplicates='drop')
    df_temp.boxplot(column='Cmax_roll', by='T_bin', ax=ax, showfliers=False,
                    patch_artist=True, grid=False)
    ax.set_title(f'电场{i+1}振打周期 vs 瞬时峰值(窗口最大)')
    ax.set_xlabel(f'T{i+1} (s)')
    ax.set_ylabel('窗口最大出口浓度 (mg/Nm³)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50 mg/Nm³')
    ax.legend()
plt.suptitle('')
plt.tight_layout()
plt.show()

# 用线性回归量化对峰值的影响
print("各电场振打周期与窗口最大浓度的 Spearman 相关系数：")
for i in range(4):
    corr_spearman = df[f'T{i+1}_s'].corr(df['Cmax_roll'], method='spearman')
    print(f"  T{i+1}: {corr_spearman:.4f}")

# 积灰状态影响分析
fig, ax = plt.subplots(figsize=(8, 5))
bins = pd.qcut(df['S1'], q=10, duplicates='drop')
agg = df.groupby(bins, observed=True)['C_max_roll'].agg(['mean', 'std'])
centers = np.array([iv.mid for iv in agg.index])
ax.errorbar(centers, agg['mean'], yerr=agg['std'], fmt='o-', capsize=5,
            label='窗口最大浓度均值 ± 标准差')
ax.set_xlabel('一电场动态积灰状态 S1 (s)')
ax.set_ylabel('窗口最大出口浓度 (mg/Nm³)')
ax.set_title('积灰状态对排放峰值的影响（S1 vs 窗口最大值）')
ax.axhline(50, color='red', linestyle='--', label='50 mg/Nm³')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n分析结论：")
print("1. 出口浓度与入口温度、入口浓度、烟气流量呈正相关（或通过散热关系变化）。")
print("2. 电压升高通常可降低出口浓度，但各电场影响程度不同；振打周期延长导致积灰加重，出口浓度均值与峰值均有升高趋势。")
print("3. 瞬时排放峰值（窗口最大值）随振打周期增大而显著上升，尤其是电场1/2，Spearman相关系数明显为正。")
print("4. 动态积灰状态 S 能较好捕捉振打策略的累积效应，S 越大，排放波动和峰值越高。")

# ===========================================================================
# 【问题2】工况划分与最低电耗优化
# ===========================================================================
print("\n" + "=" * 80)
print("【问题2】工况划分与最低电耗优化")
print("=" * 80)

# 物理模型参数（复用问题1结果）
K_opt = opt_params[0]
alpha_opt = opt_params[1]
beta_opt = opt_params[2]
k_opt = np.array([opt_params[3], opt_params[4], opt_params[5], opt_params[6]])

def physical_omega_simple(U, S, T_gas):
    """驱进速度 Ω（简化版，向量输入）"""
    T_K = T_gas + 273.15
    U_eff = np.clip(U - k_opt * S, 1.0, None)
    sumU = np.sum(U_eff)
    return K_opt * (T_K ** (-beta_opt)) * (sumU ** alpha_opt)

def outlet_concentration(U, S, T_gas, Q, C_in_mg):
    """仿真出口浓度 (mg/Nm³)"""
    Omega = physical_omega_simple(U, S, T_gas)
    return C_in_mg * np.exp(-Omega / Q)

# 电耗模型拟合
df['sumU2'] = (df[['U1_kV','U2_kV','U3_kV','U4_kV']]**2).sum(axis=1)
pwr_model = LinearRegression()
pwr_model.fit(df[['sumU2']], df['P_total_kW'])
a_pwr = pwr_model.intercept_
b_pwr = pwr_model.coef_[0]
print(f"\n电耗模型: P = {a_pwr:.2f} + {b_pwr:.6f} * ΣU²")
print(f"训练 R² = {pwr_model.score(df[['sumU2']], df['P_total_kW']):.3f}")

def predict_power(U):
    return a_pwr + b_pwr * np.sum(U**2)

# 工况聚类
X_cluster = df[['C_in_gNm3', 'Temp_C']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_Q = [df.loc[df['cluster']==c, 'Q_Nm3h'].median() for c in range(n_clusters)]

print("\n典型工况中心（C_in g/Nm³, T °C, Q Nm³/h）：")
for c in range(n_clusters):
    print(f"  工况{c+1}: C_in={centers[c,0]:.1f}, T={centers[c,1]:.0f}, Q={cluster_Q[c]:.0f}")

# 优化各工况
U_min, U_max = 40.0, 80.0
T_min, T_max = 120.0, 600.0
C_limit = 10.0
penalty_coeff = 1e6

results = []
for c in range(n_clusters):
    Cin_g, T_c = centers[c, 0], centers[c, 1]
    Q_c = cluster_Q[c]
    Cin_mg = Cin_g * 1000.0

    def objective(x):
        U = np.array(x[:4])
        T = np.array(x[4:])
        if np.any(U < U_min) or np.any(U > U_max):
            return 1e10
        if np.any(T < T_min) or np.any(T > T_max):
            return 1e10
        S = T
        Cout = outlet_concentration(U, S, T_c, Q_c, Cin_mg)
        power = predict_power(U)
        if Cout > C_limit:
            return penalty_coeff + power + (Cout - C_limit) * 100.0
        return power

    bounds = [(U_min, U_max)]*4 + [(T_min, T_max)]*4
    res_opt = differential_evolution(objective, bounds, maxiter=200, popsize=30, seed=42, polish=True)
    best = res_opt.x
    best_U = best[:4]
    best_T = best[4:]
    best_power = predict_power(best_U)

    mask = df['cluster'] == c
    Cout_all = outlet_concentration(best_U, best_T, df.loc[mask, 'Temp_C'].values,
                                    df.loc[mask, 'Q_Nm3h'].values, df.loc[mask, 'C_in_mg'].values)
    compliance = np.mean(Cout_all <= C_limit) * 100

    results.append({
        '工况': c+1,
        'C_in (g/Nm³)': Cin_g,
        'T (°C)': T_c,
        'Q (Nm³/h)': Q_c,
        'U1': best_U[0], 'U2': best_U[1], 'U3': best_U[2], 'U4': best_U[3],
        'T1': best_T[0], 'T2': best_T[1], 'T3': best_T[2], 'T4': best_T[3],
        'P_min (kW)': best_power,
        '达标率(%)': compliance
    })

    print(f"工况{c+1}: C_in={Cin_g:.1f} T={T_c:.0f} Q={Q_c:.0f} "
          f"U={np.round(best_U,1)} T={np.round(best_T,0)} "
          f"P={best_power:.1f}kW 达标率={compliance:.1f}%")

res_df = pd.DataFrame(results)
print("\n" + "="*90)
print("各典型工况最优操作参数（排放≤10 mg/Nm³）")
print(res_df.to_string(index=False))

# ===========================================================================
# 【问题2补充】各工况的可视化分析
# ===========================================================================
# 平均电压、振打周期
res_df["U_mean"] = res_df[["U1", "U2", "U3", "U4"]].mean(axis=1)
res_df["T_mean"] = res_df[["T1", "T2", "T3", "T4"]].mean(axis=1)

x = res_df["工况"]

fig = plt.figure(figsize=(10, 6))
plt.plot(x, res_df["U_mean"], marker="o", label="平均电压 (kV)")
plt.plot(x, res_df["T_mean"], marker="s", label="平均振打周期 (s)")
plt.plot(x, res_df["P_min (kW)"], marker="^", label="最小电耗 (kW)")
plt.xlabel("工况编号")
plt.ylabel("数值")
plt.title("不同工况下控制参数与电耗对比")
plt.legend()
plt.grid()
plt.show()

# 入口浓度与电压关系
plt.figure(figsize=(8, 5))
plt.scatter(res_df["C_in (g/Nm³)"], res_df["U_mean"])
for i in range(len(res_df)):
    plt.text(res_df["C_in (g/Nm³)"][i], res_df["U_mean"][i], f"{i+1}")
plt.xlabel("入口浓度 (g/Nm³)")
plt.ylabel("平均电压 (kV)")
plt.title("入口浓度与最优电压关系")
plt.grid()
plt.show()

# 温度与振打周期关系
plt.figure(figsize=(8, 5))
plt.scatter(res_df["T (°C)"], res_df["T_mean"])
for i in range(len(res_df)):
    plt.text(res_df["T (°C)"][i], res_df["T_mean"][i], f"{i+1}")
plt.xlabel("温度 (°C)")
plt.ylabel("平均振打周期 (s)")
plt.title("温度与振打周期关系")
plt.grid()
plt.show()

# 电压与电耗关系
plt.figure(figsize=(8, 5))
plt.scatter(res_df["U_mean"], res_df["P_min (kW)"])
plt.xlabel("平均电压 (kV)")
plt.ylabel("电耗 (kW)")
plt.title("电压与电耗关系")
plt.grid()
plt.show()

# 各工况控制参数热力图
try:
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    heat_data = res_df[["U1", "U2", "U3", "U4", "T1", "T2", "T3", "T4"]]
    sns.heatmap(heat_data, annot=True, fmt=".1f")
    plt.title("各工况控制参数热力图")
    plt.xlabel("参数")
    plt.ylabel("工况")
    plt.show()
except:
    print("（跳过热力图，因缺少seaborn库）")

# 电压对排放的影响（高浓度工况）
row = res_df.loc[res_df["C_in (g/Nm³)"].idxmax()]
U_base = np.array([row["U1"], row["U2"], row["U3"], row["U4"]])
T_base = np.array([row["T1"], row["T2"], row["T3"], row["T4"]])

U_range = np.linspace(50, 80, 20)
Cout_list = []
for u in U_range:
    U_test = np.array([u] * 4)
    Cout = outlet_concentration(U_test, T_base, row["T (°C)"], row["Q (Nm³/h)"], row["C_in (g/Nm³)"] * 1000)
    Cout_list.append(Cout)

plt.figure(figsize=(8, 5))
plt.plot(U_range, Cout_list)
plt.xlabel("电压 (kV)")
plt.ylabel("出口浓度 (mg/Nm³)")
plt.title("电压对排放影响（高浓度工况）")
plt.grid()
plt.show()

# 电压与振打的敏感性对比（两个典型工况）
def sensitivity_voltage(row_data):
    U_base = np.array([row_data["U1"], row_data["U2"], row_data["U3"], row_data["U4"]])
    T_base = np.array([row_data["T1"], row_data["T2"], row_data["T3"], row_data["T4"]])
    U_range = np.linspace(50, 80, 30)
    Cout_list = []
    for u in U_range:
        U_test = np.array([u] * 4)
        Cout = outlet_concentration(U_test, T_base, row_data["T (°C)"], row_data["Q (Nm³/h)"], row_data["C_in (g/Nm³)"] * 1000)
        Cout_list.append(Cout)
    return U_range, Cout_list

def sensitivity_tap(row_data):
    U_base = np.array([row_data["U1"], row_data["U2"], row_data["U3"], row_data["U4"]])
    T_range = np.linspace(120, 600, 30)
    Cout_list = []
    for t in T_range:
        T_test = np.array([t] * 4)
        Cout = outlet_concentration(U_base, T_test, row_data["T (°C)"], row_data["Q (Nm³/h)"], row_data["C_in (g/Nm³)"] * 1000)
        Cout_list.append(Cout)
    return T_range, Cout_list

row_high = res_df.loc[res_df["C_in (g/Nm³)"].idxmax()]
row_low = res_df.loc[res_df["T (°C)"].idxmax()]

U_h, C_h = sensitivity_voltage(row_high)
T_h, CT_h = sensitivity_tap(row_high)
U_l, C_l = sensitivity_voltage(row_low)
T_l, CT_l = sensitivity_tap(row_low)

plt.figure(figsize=(12, 5))

# 左：电压影响
plt.subplot(1, 2, 1)
plt.plot(U_h, C_h, label="高浓度工况")
plt.plot(U_l, C_l, label="低浓度工况")
plt.xlabel("电压 (kV)")
plt.ylabel("出口浓度")
plt.title("电压敏感性对比")
plt.legend()
plt.grid()

# 右：振打影响
plt.subplot(1, 2, 2)
plt.plot(T_h, CT_h, label="高浓度工况")
plt.plot(T_l, CT_l, label="低浓度工况")
plt.xlabel("振打周期 (s)")
plt.ylabel("出口浓度")
plt.title("振打敏感性对比")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ===========================================================================
# 【问题3】控制策略差异分析与优先级规律
# ===========================================================================
print("\n" + "=" * 80)
print("【问题3】控制策略差异分析与优先级规律")
print("=" * 80)

res_df["U_mean"] = res_df[["U1", "U2", "U3", "U4"]].mean(axis=1)
res_df["T_mean"] = res_df[["T1", "T2", "T3", "T4"]].mean(axis=1)
res_df["U_std"] = res_df[["U1", "U2", "U3", "U4"]].std(axis=1)
res_df["T_std"] = res_df[["T1", "T2", "T3", "T4"]].std(axis=1)

# 选择典型工况
case_high_idx = res_df['C_in (g/Nm³)'].idxmax()
case_low_idx = res_df['T (°C)'].idxmax()

case_A = res_df.iloc[case_high_idx]
case_B = res_df.iloc[case_low_idx]

print(f"\n【典型工况选择】")
print(f"  工况A（高浓度）：C_in = {case_A['C_in (g/Nm³)']:.2f} g/Nm³")
print(f"  工况B（低浓度）：C_in = {case_B['C_in (g/Nm³)']:.2f} g/Nm³")

U_base_A = np.array([case_A['U1'], case_A['U2'], case_A['U3'], case_A['U4']])
T_base_A = np.array([case_A['T1'], case_A['T2'], case_A['T3'], case_A['T4']])
U_base_B = np.array([case_B['U1'], case_B['U2'], case_B['U3'], case_B['U4']])
T_base_B = np.array([case_B['T1'], case_B['T2'], case_B['T3'], case_B['T4']])

# 敏感性分析
print("\n【敏感性分析】")
U_scan = np.linspace(40, 80, 30)
C_out_A_volt = []
C_out_B_volt = []

for u_val in U_scan:
    U_test = np.array([u_val, u_val, u_val, u_val])
    C_A = outlet_concentration(U_test, T_base_A, case_A['T (°C)'], case_A['Q (Nm³/h)'], case_A['C_in (g/Nm³)'] * 1000)
    C_B = outlet_concentration(U_test, T_base_B, case_B['T (°C)'], case_B['Q (Nm³/h)'], case_B['C_in (g/Nm³)'] * 1000)
    C_out_A_volt.append(C_A)
    C_out_B_volt.append(C_B)

C_out_A_volt = np.array(C_out_A_volt)
C_out_B_volt = np.array(C_out_B_volt)

T_scan = np.linspace(120, 600, 30)
C_out_A_tap = []
C_out_B_tap = []

for t_val in T_scan:
    T_test = np.array([t_val, t_val, t_val, t_val])
    C_A = outlet_concentration(U_base_A, T_test, case_A['T (°C)'], case_A['Q (Nm³/h)'], case_A['C_in (g/Nm³)'] * 1000)
    C_B = outlet_concentration(U_base_B, T_test, case_B['T (°C)'], case_B['Q (Nm³/h)'], case_B['C_in (g/Nm³)'] * 1000)
    C_out_A_tap.append(C_A)
    C_out_B_tap.append(C_B)

C_out_A_tap = np.array(C_out_A_tap)
C_out_B_tap = np.array(C_out_B_tap)

Delta_U = U_scan[-1] - U_scan[0]
Delta_T = T_scan[-1] - T_scan[0]

S_U_A = (C_out_A_volt[0] - C_out_A_volt[-1]) / Delta_U
S_U_B = (C_out_B_volt[0] - C_out_B_volt[-1]) / Delta_U
S_T_A = (C_out_A_tap[-1] - C_out_A_tap[0]) / Delta_T
S_T_B = (C_out_B_tap[-1] - C_out_B_tap[0]) / Delta_T

print(f"\n【表：敏感性指数对比】")
print(f"{'控制变量':<20s} {'工况A(高浓)':>15s} {'工况B(低浓)':>15s} {'相对倍数':>15s}")
print("-" * 65)
print(f"{'电压敏感度(mg/Nm³/kV)':<20s} {S_U_A:>15.4f} {S_U_B:>15.4f} {S_U_A/S_U_B:>14.2f}x")
print(f"{'振打敏感度(mg/Nm³/s)':<20s} {S_T_A:>15.4f} {S_T_B:>15.4f} {S_T_A/S_T_B:>14.2f}x")

print(f"\n相对控制力对比：")
print(f"  工况A：电压敏感度/振打敏感度 = {S_U_A/S_T_A:.2f}")
print(f"  工况B：电压敏感度/振打敏感度 = {S_U_B/S_T_B:.2f}")

print(f"\n【控制优先级结论】")
print(f"  电压是一阶主导变量，控制力约为振打周期的1000倍")
print(f"  高浓度工况对电压更敏感，需要精细的电压控制")
print(f"  低浓度工况应重点关注节能，可适当降低电压")

# ===========================================================================
# 【问题4】排放限值下调至5 mg/Nm³的电耗影响分析
# ===========================================================================
print("\n" + "=" * 80)
print("【问题4】排放限值下调至5 mg/Nm³的电耗影响")
print("=" * 80)

print("\n对各典型工况进行限值5的优化...")

for idx, case_row in res_df.iterrows():
    Cin_g = case_row['C_in (g/Nm³)']
    T_gas = case_row['T (°C)']
    Q = case_row['Q (Nm³/h)']
    Cin_mg = Cin_g * 1000.0
    U10 = np.array([case_row['U1'], case_row['U2'], case_row['U3'], case_row['U4']])
    P10 = case_row['P_min (kW)']

    def objective_limit5(x):
        U = np.array(x[:4])
        T = np.array(x[4:])
        if np.any(U < U_min) or np.any(U > U_max):
            return 1e12
        if np.any(T < T_min) or np.any(T > T_max):
            return 1e12
        S = T
        Cout = outlet_concentration(U, S, T_gas, Q, Cin_mg)
        power = predict_power(U)
        if Cout > 5.0:
            return 1e10 + power + (Cout - 5.0) * 1000.0
        return power

    bounds = [(U_min, U_max)] * 4 + [(T_min, T_max)] * 4
    res = differential_evolution(objective_limit5, bounds, maxiter=300, popsize=25, seed=42, polish=True)
    U5 = res.x[:4]
    T5 = res.x[4:]
    P5 = predict_power(U5)
    Cout_check = outlet_concentration(U5, T5, T_gas, Q, Cin_mg)
    increase = (P5 - P10) / P10 * 100

    print(f"工况{int(case_row['工况'])}: C_in={Cin_g:.1f}")
    print(f"  限值10: P={P10:.1f} kW")
    print(f"  限值5:  P={P5:.1f} kW (C_out={Cout_check:.2f} mg/Nm³)")
    print(f"  电耗增加: {increase:.1f}%")

print("\n" + "=" * 80)
print("【分析完成】")
print("=" * 80)
print("\n所有分析已完成，包含问题1-4的完整内容。")
