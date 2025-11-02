import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = "data/retail_sales_sample.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA, parse_dates=["Date"]) #parse_dates=["Date"] tells Pandas to convert the Date column to real date/time values (not plain text).
print("Loaded:", df.shape, "rows,cols")

# 2. Quick checks & simple cleaning
print(df.info())
print("Missing values:\n", df.isnull().sum())
df = df.drop_duplicates()
numeric = ["Quantity", "Price", "Sales", "Discount", "Profit"]

for c in numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

# 3. Feature engineering (useful business columns)
df["NetSales"] = (df["Sales"] * (1 - df["Discount"])).round(2)
# Avoid division by zero
df["ProfitMarginPct"] = np.where(df["NetSales"] > 0,
                                 (df["Profit"] / df["NetSales"]) * 100,
                                 0).round(2)
df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.month_name()

# 4. Simple analysis (easy-to-read results)
total_netsales = df["NetSales"].sum()
total_profit = df["Profit"].sum()
avg_margin = df["ProfitMarginPct"].mean()

sales_by_cat = df.groupby("Category")["NetSales"].sum().sort_values(ascending=False)
top_products = df.groupby("Product")["NetSales"].sum().sort_values(ascending=False).head(8)
region_perf = df.groupby("Region")[["NetSales", "Profit"]].sum().sort_values("NetSales", ascending=False)

print("\n--- Key KPIs ---")
print("Total Net Sales:", round(total_netsales, 2))
print("Total Profit:", round(total_profit, 2))
print("Avg Profit Margin (%):", round(avg_margin, 2))
print("\nTop categories:\n", sales_by_cat)
print("\nTop products (top 8):\n", top_products)
print("\nRegion performance:\n", region_perf)

# 5. Save cleaned data and a small summary
df.to_csv(os.path.join(OUT, "retail_sales_cleaned.csv"), index=False)
summary = {
    "total_netsales": total_netsales,
    "total_profit": total_profit,
    "avg_profit_margin": avg_margin,
    "top_category": sales_by_cat.index[0] if not sales_by_cat.empty else "",
    "top_product": top_products.index[0] if not top_products.empty else ""
}
pd.Series(summary).to_frame("value").to_csv(os.path.join(OUT, "summary_metrics.csv"))
print("\nSaved cleaned data and summary in", OUT)

# 6. Two simple plots
plt.figure(figsize=(8,4))
sales_by_cat.plot(kind="bar", legend=False)
plt.title("Net Sales by Category")
plt.ylabel("Net Sales")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_sales_by_category.png"))
plt.close()

plt.figure(figsize=(8,4))
top_products.sort_values().plot(kind="barh", legend=False)
plt.title("Top Products by Net Sales")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_top_products.png"))
plt.close()

print("Saved plots in", OUT)
print("Done — open outputs/ to see cleaned data, summary, and plots.")
