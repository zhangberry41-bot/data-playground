"""
pandas 入门 demo：分析一家模拟奶茶店一周的销售数据。

运行：
    pip install -r requirements.txt
    python pandas_demo.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PRODUCTS = ["珍珠奶茶", "杨枝甘露", "柠檬水", "芋泥波波", "烤奶"]
PRICES = {"珍珠奶茶": 12, "杨枝甘露": 18, "柠檬水": 9, "芋泥波波": 15, "烤奶": 13}
DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def make_orders() -> pd.DataFrame:
    """生成一周的模拟订单数据。"""
    rows = []
    order_id = 1
    for day in DAYS:
        # 周末单量更多
        low, high = (120, 200) if day in ("周六", "周日") else (80, 140)
        for _ in range(rng.integers(low, high)):
            product = rng.choice(PRODUCTS)
            qty = int(rng.integers(1, 4))
            rows.append((order_id, day, product, qty, PRICES[product] * qty))
            order_id += 1
    return pd.DataFrame(rows, columns=["订单号", "星期", "产品", "数量", "金额"])


def main() -> None:
    df = make_orders()
    print(f"一周共 {len(df)} 笔订单")

    # 1. 总营收 & 客单价
    total = df["金额"].sum()
    print(f"总营收：{total} 元，平均客单价：{total / len(df):.1f} 元")

    # 2. 哪个产品最赚钱
    by_product = df.groupby("产品")["金额"].sum().sort_values(ascending=False)
    print("\n各产品营收排名：")
    print(by_product.to_string())

    # 3. 哪天生意最好
    by_day = df.groupby("星期")["金额"].sum().reindex(DAYS)
    print("\n每日营收：")
    print(by_day.to_string())
    print(f"\n生意最好的是：{by_day.idxmax()}（{by_day.max()} 元）")

    # 4. 销量最高的产品卖了多少杯
    by_qty = df.groupby("产品")["数量"].sum().sort_values(ascending=False)
    top = by_qty.index[0]
    print(f"\n销量冠军：{top}，一周卖出 {by_qty.iloc[0]} 杯")


if __name__ == "__main__":
    main()
