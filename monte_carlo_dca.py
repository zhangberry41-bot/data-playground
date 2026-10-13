"""
蒙特卡洛模拟：定投 vs 一次性买入，到底谁更强？

假设某宽基指数年化收益 8%、年化波动 20%（几何布朗运动），
本金都是 12 万，期限 10 年：
  - 策略 A：开局一次性全仓买入
  - 策略 B：每月定投 1000 元，分 120 个月投完

跑 5000 次模拟，对比两种策略最终资产的中位数、均值和亏损概率。

运行：
    pip install -r requirements.txt
    python monte_carlo_dca.py
输出：
    dca_vs_lump.png（对比图）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

ANNUAL_RETURN = 0.08   # 年化收益
ANNUAL_VOL = 0.20      # 年化波动
YEARS = 10
MONTHS = YEARS * 12
TOTAL = 120_000        # 总本金
MONTHLY = 1_000        # 每月定投额
N_SIM = 5000

mu_m = ANNUAL_RETURN / 12
sigma_m = ANNUAL_VOL / np.sqrt(12)


def simulate() -> tuple[np.ndarray, np.ndarray]:
    """返回两种策略每次模拟的期末资产。"""
    # 每月收益率：5000 次模拟 x 120 个月
    r = rng.normal(mu_m, sigma_m, size=(N_SIM, MONTHS))
    growth = 1 + r

    # 策略 A：一次性买入，持有 120 个月
    lump = TOTAL * growth.prod(axis=1)

    # 策略 B：每月定投 1000，第 k 个月投入的钱增长 (120-k) 个月
    cum_growth = np.cumprod(growth[:, ::-1], axis=1)[:, ::-1]
    dca = (MONTHLY * cum_growth).sum(axis=1)
    return lump, dca


def main() -> None:
    lump, dca = simulate()
    df = pd.DataFrame({"一次性买入": lump, "每月定投": dca})

    print("=== 5000 次模拟 · 10 年 · 本金 12 万 ===")
    print(df.describe().loc[["mean", "50%", "min", "max"]].round(0).to_string())
    for col in df.columns:
        loss_prob = (df[col] < TOTAL).mean() * 100
        print(f"{col}：亏损概率 {loss_prob:.1f}%")
    print(f"\n定投跑赢一次性买入的概率：{(dca > lump).mean() * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：示例路径
    months = np.arange(1, MONTHS + 1)
    r = rng.normal(mu_m, sigma_m, size=(3, MONTHS))
    price = 100 * np.cumprod(1 + r, axis=1)
    for i in range(3):
        axes[0].plot(months, price[i], alpha=0.8)
    axes[0].set_title("指数走势示例（3 次模拟）", fontproperties=FONT)
    axes[0].set_xlabel("月份", fontproperties=FONT)
    axes[0].set_ylabel("指数点位", fontproperties=FONT)

    # 右：期末资产分布
    axes[1].hist(lump / 1e4, bins=60, alpha=0.6, label="一次性买入")
    axes[1].hist(dca / 1e4, bins=60, alpha=0.6, label="每月定投")
    axes[1].axvline(TOTAL / 1e4, color="red", linestyle="--", label="本金 12 万")
    axes[1].set_title("10 年后资产分布（万元）", fontproperties=FONT)
    axes[1].set_xlabel("期末资产（万元）", fontproperties=FONT)
    axes[1].legend(prop=FONT)

    fig.tight_layout()
    fig.savefig("dca_vs_lump.png", dpi=120)
    print("\n对比图已保存：dca_vs_lump.png")


if __name__ == "__main__":
    main()
