# %% [markdown]
# # UK confectionery sales: cleaning a messy dataset, then finding where the profit is
#
# **Brief:** a confectionery council wants to improve regional distribution and gross profit. Which products have the largest and smallest margins in each region, when do regions peak, and how do regions compare?
#
# **Data:** 1,001 sales records (2000–2005) across five UK regions and six products, with units sold, cost, profit and revenue.
#
# Most of the work here is **cleaning**: the raw file has misspelt products, mislabelled rows, missing values and a broken revenue column. Each fix is checked against the data before it is applied.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import style
from common.style import BLUE, ORANGE, AQUA, MUTED, SERIES

style.apply()
raw = pd.read_excel("data/confectionery_sales.xlsx")
raw.columns = ["date", "region", "product", "units", "cost", "profit", "revenue"]
print(raw.shape)
raw.head()

# %% [markdown]
# ## 1. Data quality problems

# %%
print("Missing values:", raw.isna().sum()[lambda s: s > 0].to_dict())
print("Negative revenue rows:", (raw["revenue"] < 0).sum(), "| zero revenue rows:", (raw["revenue"] == 0).sum())
raw["product"].value_counts()

# %% [markdown]
# **Problem 1: inconsistent spelling.** "Choclate Chunk" / "Chocolate Chunk" and "Caramel nut" / "Caramel Nut" are the same products.

# %%
df = raw.copy()
df["product"] = (df["product"].str.strip().str.replace("Choclate", "Chocolate", regex=False)
                 .str.replace("Caramel nut", "Caramel Nut", regex=False))
df["product"].value_counts()

# %% [markdown]
# **Problem 2: mislabelled products.** Every product has a fixed cost and profit per unit, except "Caramel", which shows five different unit costs:

# %%
df["unit_cost"] = (df["cost"] / df["units"]).round(2)
df["unit_profit"] = (df["profit"] / df["units"]).round(2)
pd.crosstab(df["product"], df["unit_cost"])

# %% [markdown]
# The odd "Caramel" unit costs (£0.20, £1.25, £1.50, £2.20, £2.75) are exactly the unit costs of the *other* products, and they occur only in **England**. These are other products recorded as "Caramel". Since each unit cost belongs to exactly one product, the true product can be recovered:

# %%
price_book = (df[df["product"] != "Caramel"].dropna(subset=["unit_cost"])
              .groupby("product")[["unit_cost", "unit_profit"]].agg(lambda s: s.mode()[0]))
price_book.loc["Caramel"] = [2.00, 3.00]
by_cost = {round(c, 2): p for p, c in price_book["unit_cost"].items()}

wrong = (df["product"] == "Caramel") & df["unit_cost"].notna() & (df["unit_cost"] != 2.00)
print(f"{wrong.sum()} rows relabelled, all in: {df.loc[wrong, 'region'].unique().tolist()}")
df.loc[wrong, "product"] = df.loc[wrong, "unit_cost"].map(by_cost)
price_book

# %% [markdown]
# **Problem 3: missing values.** With fixed unit prices, a missing cost, profit or units value can be rebuilt from the other columns.

# %%
missing_before = df[["units", "cost", "profit"]].isna().sum().sum()
uc = df["product"].map(price_book["unit_cost"])
up = df["product"].map(price_book["unit_profit"])
df["units"] = df["units"].fillna((df["cost"] / uc).round()).fillna((df["profit"] / up).round())
df["cost"] = df["cost"].fillna(df["units"] * uc)
df["profit"] = df["profit"].fillna(df["units"] * up)
print(f"Filled {missing_before} missing values; remaining: {df[['units', 'cost', 'profit']].isna().sum().sum()}")

# %% [markdown]
# **Problem 4: the revenue column uses the wrong formula.** It equals units × (profit − cost), which is why some values are negative or zero:

# %%
formula = df["units"] * (df["profit"] - df["cost"])
match = np.isclose(formula, raw["revenue"], rtol=1e-6)
print(f"revenue == units × (profit − cost) for {match.mean():.0%} of rows")
df["sales"] = df["cost"] + df["profit"]      # true sales value
df["margin"] = df["profit"] / df["sales"]
df = df.drop(columns=["revenue", "unit_cost", "unit_profit"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df.head()

# %% [markdown]
# ## 2. Which products are most profitable?

# %%
products = df.groupby("product").agg(units=("units", "sum"), sales=("sales", "sum"), profit=("profit", "sum"))
products["margin_%"] = 100 * products["profit"] / products["sales"]
products["profit_per_unit"] = price_book["unit_profit"]
products = products.sort_values("profit", ascending=False)
products.round(1)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
style.hbar(axes[0], products.index, products["profit"] / 1e6, fmt="£{:.2f}M")
axes[0].set_title("Total profit 2000–2005")
m = products.sort_values("margin_%", ascending=False)
style.hbar(axes[1], m.index, m["margin_%"], fmt="{:.1f}%", color=AQUA)
axes[1].set_title("Profit margin (profit ÷ sales)")
fig.tight_layout()
style.save(fig, "product_profit_margin")
plt.show()

# %% [markdown]
# Because prices are fixed, each product's margin is the same in every region. The brief's question "largest and smallest margin per region" therefore has the same answer everywhere, so the more useful view is where each product **earns** its profit:

# %%
region_product = df.pivot_table(index="region", columns="product", values="profit", aggfunc="sum").fillna(0)
share = region_product.div(region_product.sum(axis=1), axis=0).mul(100).round(1)
share

# %%
fig, ax = plt.subplots(figsize=(10, 3.8))
left = np.zeros(len(share))
order = products.index.tolist()
for prod, color in zip(order, SERIES):
    ax.barh(share.index, share[prod], left=left, color=color, label=prod, height=0.65, edgecolor="#fcfcfb", linewidth=1.5)
    left += share[prod].values
ax.set_xlim(0, 100)
ax.set_xlabel("% of the region's profit")
ax.set_title("Profit mix by region")
ax.legend(loc="lower left", ncols=6, bbox_to_anchor=(0, 1.08), fontsize=9)
ax.grid(axis="y", visible=False)
style.save(fig, "region_profit_mix")
plt.show()

# %% [markdown]
# ## 3. How do regions compare, and when do they peak?

# %%
regions = df.groupby("region").agg(records=("date", "size"), units=("units", "sum"), profit=("profit", "sum"))
regions["profit_per_record"] = regions["profit"] / regions["records"]
regions.sort_values("profit", ascending=False).round(0)

# %%
yearly = df.pivot_table(index="year", columns="region", values="profit", aggfunc="sum") / 1e3
fig, ax = plt.subplots(figsize=(9.5, 4.3))
for reg, color in zip(regions.sort_values("profit", ascending=False).index, SERIES):
    ax.plot(yearly.index, yearly[reg], "-o", color=color, lw=2, label=reg)
    peak = yearly[reg].idxmax()
    ax.plot(peak, yearly.loc[peak, reg], "o", ms=12, mfc="none", mec=color, mew=2)
ax.set_ylabel("Profit (£ thousands)")
ax.set_title("Yearly profit by region (rings = each region's peak year)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
style.save(fig, "yearly_profit_by_region")
plt.show()
peaks = pd.DataFrame({"peak_year": yearly.idxmax(), "peak_profit_k": yearly.max().round(0),
                      "lowest_year": yearly.idxmin()})
peaks

# %%
monthly = df.groupby("month")["profit"].sum() / df["year"].nunique() / 1e3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(monthly.index, monthly.values, color=[ORANGE if m in monthly.nlargest(3).index else BLUE for m in monthly.index])
ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("Avg profit per year (£k)")
ax.set_title("Seasonality: average profit by month (top 3 in orange)")
style.save(fig, "seasonality")
plt.show()

# %% [markdown]
# ## Key findings and recommendations

# %%
top, low_margin = products.index[0], products["margin_%"].idxmin()
best_region = regions["profit"].idxmax()
eng = regions.loc["England"]
print(f"1. Data quality: {wrong.sum()} England rows were mislabelled as Caramel, {missing_before} values were missing "
      f"and the revenue column used a wrong formula; all were corrected before analysis.")
print(f"2. {top} earns the most profit (£{products.loc[top, 'profit'] / 1e6:.2f}M) thanks to volume and a high "
      f"£{products.loc[top, 'profit_per_unit']:.2f} profit per unit.")
print(f"3. Plain has the best margin ({products.loc['Plain', 'margin_%']:.0f}%) but earns only "
      f"£{products.loc['Plain', 'profit_per_unit']:.2f} per unit; {low_margin} has the lowest margin "
      f"({products.loc[low_margin, 'margin_%']:.0f}%) but the highest profit per unit (£{products.loc[low_margin, 'profit_per_unit']:.2f}). "
      "Margin alone is the wrong target.")
rank = regions["profit_per_record"].rank(ascending=False).astype(int)
print(f"4. {best_region} is the most profitable region (£{regions.loc[best_region, 'profit'] / 1e6:.2f}M). England ranks last on "
      f"total profit only because it has fewer records ({eng['records']:.0f} vs 210 elsewhere): its profit per sale "
      f"(£{eng['profit_per_record']:,.0f}) ranks #{rank['England']} of 5.")
print(f"5. Peak years differ by region ({', '.join(f'{r} {y}' for r, y in peaks['peak_year'].items())}), and the best "
      f"months overall are {', '.join(pd.to_datetime(monthly.nlargest(3).index, format='%m').strftime('%B'))}.")
print("Recommendations: prioritise distribution of Caramel and Caramel Nut, which drive profit per unit; investigate why "
      "England has 23% fewer records than every other region (missing data or a real distribution gap); "
      "fix the revenue formula and product coding at the source.")
