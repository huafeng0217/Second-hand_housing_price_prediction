#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ames Housing 数据集全流程数据处理脚本
==============================================
包含：数据清洗 → 缺失值处理 → 异常值检测 → 目标变换 → EDA可视化 → 最终交付
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# 0. 全局设置
# ============================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# 尝试设置中文字体，若失败则回退
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

report_lines = []  # 收集统计报告行


def log(msg):
    print(msg)
    report_lines.append(msg)


# ============================================================
# 1. 数据加载
# ============================================================
log("=" * 70)
log(f"Ames Housing 全流程数据处理 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

DATA_PATH = "AmesHousing.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)
log(f"\n[加载] 原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 2. 数据清洗
# ============================================================

# --- 2.1 重复值 ---
n_dup = df.duplicated().sum()
log(f"\n[清洗] 完全重复行: {n_dup}")
if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"       → 已删除重复行，当前 {df.shape[0]} 行")

# --- 2.2 无效列过滤 ---
# 'Order' 和 'PID' 是纯标识符，不参与建模
id_cols = ["Order", "PID"]
df = df.drop(columns=id_cols)
log(f"\n[清洗] 已移除标识列: {id_cols}")

# --- 2.3 把 CSV 中的字符串 'NA' 转为真正的 NaN ---
# AmesHousing.csv 中缺失的分类值写作 "NA"，需与 numpy NaN 区分
na_like_cols = []
for col in df.columns:
    if df[col].dtype == object:
        mask = df[col].isin(["NA", "na", "N/A", ""])
        if mask.any():
            na_like_cols.append(col)
            df.loc[mask, col] = np.nan

log(f"\n[清洗] 将 'NA'/'N/A'/空串 转为 NaN 的列: {len(na_like_cols)} 列")

# --- 2.4 数据类型修正 ---
# 识别各类列的合理类型
numerical_cols = []
categorical_cols = []
ordinal_num_cols = []  # 用数字编码但本质是分类/有序的列

# 基于 Ames 数据字典手动划分
numeric_candidates = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area",
    "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
    "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
    "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch",
    "Screen Porch", "Pool Area", "Misc Val",
    "Mo Sold", "Yr Sold", "SalePrice",
]
# 数字编码但应视为有序分类的列
ordinal_candidates = [
    "MS SubClass", "Overall Qual", "Overall Cond",
]

for col in df.columns:
    if col in numeric_candidates:
        numerical_cols.append(col)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif col in ordinal_candidates:
        ordinal_num_cols.append(col)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        categorical_cols.append(col)
        df[col] = df[col].astype("category")

log(f"\n[类型] 连续数值型: {len(numerical_cols)} 列")
log(f"       有序数值型(保留为数值): {len(ordinal_num_cols)} 列")
log(f"       分类型:       {len(categorical_cols)} 列")

# ============================================================
# 3. 缺失值处理
# ============================================================

log("\n" + "=" * 70)
log("缺失值统计与分析")
log("=" * 70)

# --- 3.1 缺失值分布统计 ---
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "列名": missing_counts.index,
    "缺失数": missing_counts.values,
    "缺失率(%)": missing_pct.values,
})
missing_df = missing_df[missing_df["缺失数"] > 0].sort_values("缺失数", ascending=False)
log(f"\n有缺失值的列: {len(missing_df)} / {df.shape[1]}")
log(missing_df.to_string(index=False))

# --- 3.2 缺失值标记列 ---
# 对有缺失的数值列创建缺失指示变量
marked_cols = []
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        marker_name = col + "_missing"
        df[marker_name] = df[col].isnull().astype(int)
        marked_cols.append(marker_name)

log(f"\n[缺失标记] 已为 {len(marked_cols)} 个数值列创建缺失指示变量")

# --- 3.3 数值型缺失值填充（中位数） ---
num_missing_filled = []
for col in numerical_cols:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        num_missing_filled.append((col, n_miss, median_val))
        log(f"  [数值填充] {col}: {n_miss} 个缺失 → 中位数 {median_val:.1f}")

# --- 3.4 分类型缺失值填充（众数 + "Missing"类别） ---
cat_missing_filled = []
for col in categorical_cols + ordinal_num_cols:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        if col in ordinal_num_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            cat_missing_filled.append((col, n_miss, f"中位数 {median_val:.1f}"))
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].cat.add_categories("Missing")
                df[col] = df[col].fillna("Missing")
                cat_missing_filled.append((col, n_miss, f"众数 '{mode_val[0]}' + 新增类别 'Missing'"))

for col, n, strategy in cat_missing_filled:
    log(f"  [分类填充] {col}: {n} 个缺失 → {strategy}")

log(f"\n[缺失处理完成] 剩余缺失值总数: {df.isnull().sum().sum()}")

# ============================================================
# 4. 异常值检测与处理
# ============================================================

log("\n" + "=" * 70)
log("异常值检测 (IQR + Z-Score 双方法)")
log("=" * 70)

# --- 4.1 定义检测函数 ---
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper))
    return outliers, lower, upper


def detect_outliers_zscore(series, threshold=3.0):
    z = np.abs((series - series.mean()) / series.std())
    return z > threshold, threshold


# --- 4.2 检测所有数值列 ---
# 选择参与异常值检测的数值列（排除年份、月份、面积类累计列）
outlier_check_cols = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch",
    "Screen Porch", "Pool Area", "Misc Val", "SalePrice",
]

outlier_summary = []

for col in outlier_check_cols:
    if col not in df.columns:
        continue
    series = df[col].dropna()
    if len(series) == 0:
        continue

    # IQR 方法
    iqr_mask, iqr_low, iqr_up = detect_outliers_iqr(series)
    n_iqr = iqr_mask.sum()

    # Z-Score 方法
    z_mask, z_thresh = detect_outliers_zscore(series, threshold=3.0)
    n_z = z_mask.sum()

    # 双方法交集（更保守的判定）
    both_mask = iqr_mask & z_mask
    n_both = both_mask.sum()

    outlier_summary.append({
        "列名": col,
        "IQR下界": iqr_low,
        "IQR上界": iqr_up,
        "IQR异常数": n_iqr,
        "IQR异常率%": round(n_iqr / len(series) * 100, 2),
        "Z-Score异常数": n_z,
        "双方法异常数": n_both,
        "双方法异常率%": round(n_both / len(series) * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_summary)
log("\n[异常值统计]")
log(outlier_df.to_string(index=False))

# --- 4.3 重点目标变量 SalePrice ---
log("\n--- SalePrice 专项异常值分析 ---")
sp = df["SalePrice"]
sp_iqr_mask, sp_low, sp_up = detect_outliers_iqr(sp)
sp_z_mask, _ = detect_outliers_zscore(sp, threshold=3.0)
sp_both = sp_iqr_mask & sp_z_mask
log(f"  SalePrice IQR 异常:  {sp_iqr_mask.sum()} 条")
log(f"  SalePrice Z-Score 异常: {sp_z_mask.sum()} 条")
log(f"  双方法交集异常:        {sp_both.sum()} 条 ({sp_both.sum()/len(sp)*100:.2f}%)")
log(f"  SalePrice 范围: [{sp.min():.0f}, {sp.max():.0f}]")

# --- 4.4 温和截断处理 (Winsorization) ---
# 对 SalePrice 使用 1st 和 99th 百分位数截断
log("\n[温和截断] 对 SalePrice 按 1st/99th 百分位做截断:")
sp_p01 = df["SalePrice"].quantile(0.01)
sp_p99 = df["SalePrice"].quantile(0.99)
log(f"  1st 百分位: {sp_p01:.0f}")
log(f"  99th 百分位: {sp_p99:.0f}")

n_clipped_low = (df["SalePrice"] < sp_p01).sum()
n_clipped_high = (df["SalePrice"] > sp_p99).sum()
df["SalePrice"] = df["SalePrice"].clip(lower=sp_p01, upper=sp_p99)
log(f"  低位截断: {n_clipped_low} 条 → {sp_p01:.0f}")
log(f"  高位截断: {n_clipped_high} 条 → {sp_p99:.0f}")
log(f"  截断后范围: [{df['SalePrice'].min():.0f}, {df['SalePrice'].max():.0f}]")

# 对其他关键数值列也做温和截断（1st/99th）
clip_cols = ["Lot Area", "Gr Liv Area", "Total Bsmt SF", "1st Flr SF"]
for col in clip_cols:
    if col not in df.columns:
        continue
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    n_high = (df[col] > p99).sum()
    n_low = (df[col] < p01).sum()
    if n_low + n_high > 0:
        df[col] = df[col].clip(lower=p01, upper=p99)
        log(f"  {col}: 低位 {n_low} / 高位 {n_high} → 已截断至 [{p01:.1f}, {p99:.1f}]")

# ============================================================
# 5. 目标变量变换
# ============================================================

log("\n" + "=" * 70)
log("目标变量: SalePrice → LogSalePrice 自然对数变换")
log("=" * 70)

# 确保 SalePrice > 0
assert (df["SalePrice"] > 0).all(), "SalePrice 存在非正值，无法对数变换"

df["LogSalePrice"] = np.log(df["SalePrice"])

original_skew = df["SalePrice"].skew()
log_skew = df["LogSalePrice"].skew()
log(f"  SalePrice 偏度:     {original_skew:.4f}")
log(f"  LogSalePrice 偏度:  {log_skew:.4f}")
log(f"  变换后峰度:         {df['LogSalePrice'].kurtosis():.4f}")

# ============================================================
# 6. EDA 可视化
# ============================================================

log("\n" + "=" * 70)
log("EDA 可视化")
log("=" * 70)

# --- 6.1 缺失值热图 ---
log("\n[绘图] 缺失值热图...")
fig, ax = plt.subplots(figsize=(18, 10))
# 只显示有缺失的列
cols_with_na = df.columns[df.isnull().any()].tolist()
if len(cols_with_na) > 0:
    # 采样展示（最多 50 列，随机 1000 行）
    sample_cols = cols_with_na[:50]
    sample_df = df[sample_cols].iloc[: min(1000, len(df))]
    sns.heatmap(
        sample_df.isnull(),
        cbar=True,
        cmap="viridis",
        xticklabels=True,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Missing Value Heatmap (sample ≤1000 rows)", fontsize=14, weight="bold")
else:
    ax.text(0.5, 0.5, "No Missing Values Remaining", ha="center", va="center", fontsize=16)
    ax.set_title("Missing Value Heatmap")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_missing_heatmap.png"), dpi=300)
plt.close(fig)
log("  → 01_missing_heatmap.png")

# --- 6.2 目标变量分布对比图 ---
log("[绘图] 目标变量分布对比图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始 SalePrice
axes[0].hist(df["SalePrice"], bins=80, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(df["SalePrice"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {df['SalePrice'].median():.0f}")
axes[0].set_title(f"SalePrice (Original)\nSkewness: {original_skew:.3f}", weight="bold")
axes[0].set_xlabel("SalePrice ($)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# 对数变换 LogSalePrice
axes[1].hist(df["LogSalePrice"], bins=80, color="forestgreen", edgecolor="white", alpha=0.85)
axes[1].axvline(df["LogSalePrice"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {df['LogSalePrice'].median():.3f}")
axes[1].set_title(f"LogSalePrice (ln-transformed)\nSkewness: {log_skew:.3f}", weight="bold")
axes[1].set_xlabel("ln(SalePrice)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_saleprice_distribution.png"), dpi=300)
plt.close(fig)
log("  → 02_saleprice_distribution.png")

# --- 6.3 箱线图 — 关键数值变量 ---
log("[绘图] 关键数值变量箱线图...")
box_cols = [
    "Lot Area", "Gr Liv Area", "Total Bsmt SF", "1st Flr SF",
    "2nd Flr SF", "Garage Area", "Wood Deck SF", "Open Porch SF",
]
box_cols = [c for c in box_cols if c in df.columns]

n_cols = 4
n_rows = int(np.ceil(len(box_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(box_cols):
    data = df[col].dropna()
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", alpha=0.7),
                    medianprops=dict(color="red", linewidth=1.5))
    axes[i].set_title(col, fontsize=10, weight="bold")
    axes[i].set_ylabel("Value")
    axes[i].tick_params(axis="x", bottom=False, labelbottom=False)

# 隐藏多余子图
for j in range(len(box_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Key Numerical Features — Boxplot", fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_boxplots.png"), dpi=300)
plt.close(fig)
log("  → 03_boxplots.png")

# --- 6.4 数值变量相关性热图 ---
log("[绘图] 相关性热图...")
# 选取数值变量（含缺失标记和 LogSalePrice）
num_cols_for_corr = [
    c for c in df.columns
    if df[c].dtype in [np.float64, np.int64, np.int32]
    and df[c].nunique() > 2
]
# 控制热图规模，选最重要的列
priority_cols = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area",
    "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
    "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
    "Fireplaces", "Garage Cars", "Garage Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "Screen Porch",
    "Pool Area", "Misc Val", "Overall Qual", "Overall Cond",
    "SalePrice", "LogSalePrice",
]
corr_cols = [c for c in priority_cols if c in df.columns and df[c].nunique() > 2]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    square=True,
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Numeric Feature Correlation Heatmap", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png"), dpi=300)
plt.close(fig)
log("  → 04_correlation_heatmap.png")

# --- 6.5 与 SalePrice 高相关特征条形图 ---
log("[绘图] SalePrice 高相关特征条形图...")
# 使用 Pearson 相关性
target_corr = corr_matrix["SalePrice"].drop("SalePrice").drop("LogSalePrice", errors="ignore")
target_corr = target_corr.sort_values(key=abs, ascending=False)
top_n = 15
top_corr = target_corr.head(top_n)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#2c7fb8" if v > 0 else "#d7191c" for v in top_corr.values]
bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with SalePrice")
ax.set_title(f"Top {top_n} Features Correlated with SalePrice", fontsize=13, weight="bold")

# 在条形末端标注数值
for bar, val in zip(bars, top_corr.values):
    x_pos = bar.get_width()
    ax.text(x_pos + 0.01 * (1 if x_pos >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_saleprice_corr_bar.png"), dpi=300)
plt.close(fig)
log("  → 05_saleprice_corr_bar.png")

# ============================================================
# 7. 最终交付
# ============================================================

log("\n" + "=" * 70)
log("最终交付")
log("=" * 70)

# --- 7.1 保存干净数据集 ---
output_csv = "AmesHousing_Cleaned.csv"
df.to_csv(output_csv, index=False)
log(f"\n[交付] 清洗后数据集: {output_csv}")
log(f"       大小: {df.shape[0]} 行 × {df.shape[1]} 列")
log(f"       文件大小: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")

# --- 7.2 数据集概览 ---
log("\n[数据集概览]")
log(f"  总行数:     {df.shape[0]}")
log(f"  总列数:     {df.shape[1]} (含缺失标记列 + LogSalePrice)")
log(f"  数值列:     {len([c for c in df.columns if df[c].dtype in [np.float64, np.int64, np.int32]])}")
log(f"  分类列:     {len([c for c in df.columns if df[c].dtype.name == 'category'])}")
log(f"  缺失标记列: {len(marked_cols)}")
log(f"  剩余缺失值: {df.isnull().sum().sum()}")
log(f"  SalePrice 截断后范围: [{df['SalePrice'].min():.0f}, {df['SalePrice'].max():.0f}]")
log(f"  LogSalePrice 范围:    [{df['LogSalePrice'].min():.4f}, {df['LogSalePrice'].max():.4f}]")

# --- 7.3 输出文件清单 ---
log("\n[输出文件]")
log(f"  {output_csv}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    log(f"  {os.path.join(OUTPUT_DIR, f)}")

# --- 7.4 保存统计报告 ---
report_path = os.path.join(OUTPUT_DIR, "processing_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n[报告] 统计报告已保存至: {report_path}")

log("\n" + "=" * 70)
log("全流程处理完毕！")
log("=" * 70)
