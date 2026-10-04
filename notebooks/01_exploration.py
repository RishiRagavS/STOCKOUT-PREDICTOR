# ============================================================
# 01_exploration.py  —  Stockout Predictor EDA
# Open in VS Code with Jupyter extension OR convert:
#   pip install jupytext
#   jupytext --to notebook 01_exploration.py
# ============================================================

# %% [markdown]
# # Stockout Predictor — Exploratory Data Analysis
#
# Covers:
# 1. Sales distribution across categories and stores
# 2. Day-of-week and seasonal demand patterns
# 3. SNAP day and holiday effects
# 4. Stockout rate analysis (simulated inventory)
# 5. Class imbalance visualization
# 6. Feature correlation heatmap
# 7. Post-training evaluation plots

# %% Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"]     = 120

PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "saved_models")

os.makedirs(MODELS, exist_ok=True)

# %% Load data
sales = pd.read_csv(os.path.join(PROCESSED, "sales_long.csv"), parse_dates=["date"])
inv   = pd.read_csv(os.path.join(PROCESSED, "inventory_snapshots.csv"), parse_dates=["date"])
feats = pd.read_csv(os.path.join(PROCESSED, "features_final.csv"), parse_dates=["date"])

print(f"sales : {sales.shape}")
print(f"inv   : {inv.shape}")
print(f"feats : {feats.shape}")

# %% [markdown]
# ## Sales Distribution by Category
# %%
cat_sales = (
    sales.groupby("cat_id")["sales"]
         .agg(["mean","median","sum"])
         .reset_index()
         .sort_values("sum", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(cat_sales["cat_id"], cat_sales["mean"],
            color=["#4C72B0","#DD8452","#55A868"])
axes[0].set_title("Average Daily Sales per Category")
axes[0].set_ylabel("Avg Units / Day")

axes[1].bar(cat_sales["cat_id"], cat_sales["sum"]/1e6,
            color=["#4C72B0","#DD8452","#55A868"])
axes[1].set_title("Total Sales Volume (Millions)")
axes[1].set_ylabel("Units (M)")

plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_category_sales.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Day-of-Week Demand Pattern
# %%
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sales["weekday"] = pd.Categorical(sales["weekday"], categories=day_order, ordered=True)
dow = sales.groupby("weekday")["sales"].mean().reset_index()

plt.figure(figsize=(10, 4))
colors = ["#4C72B0"]*5 + ["#DD8452","#DD8452"]
plt.bar(dow["weekday"].astype(str), dow["sales"], color=colors)
plt.title("Average Sales by Day of Week  (weekends highlighted)")
plt.ylabel("Avg Units / Day")
plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_day_of_week.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Holiday & SNAP Day Effect
# %%
sales["is_holiday"] = sales["event_name_1"].notna().astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

hol = sales.groupby("is_holiday")["sales"].mean().reset_index()
hol["label"] = hol["is_holiday"].map({0:"Regular Day", 1:"Holiday/Event"})
axes[0].bar(hol["label"], hol["sales"], color=["#4C72B0","#DD8452"])
axes[0].set_title("Avg Sales: Holiday vs Regular")
axes[0].set_ylabel("Avg Units / Day")

snap = sales.groupby("is_snap")["sales"].mean().reset_index()
snap["label"] = snap["is_snap"].map({0:"Non-SNAP", 1:"SNAP Day"})
axes[1].bar(snap["label"], snap["sales"], color=["#4C72B0","#55A868"])
axes[1].set_title("Avg Sales: SNAP vs Non-SNAP")
axes[1].set_ylabel("Avg Units / Day")

plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_holiday_snap.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Stockout Rate Analysis
# %%
overall = inv["stockout"].mean()
print(f"Overall stockout rate: {overall:.2%}")

store_so = (
    inv.groupby("store_id")["stockout"]
       .mean().reset_index()
       .sort_values("stockout", ascending=False)
)

plt.figure(figsize=(12, 4))
plt.bar(store_so["store_id"], store_so["stockout"]*100, color="#DD8452")
plt.axhline(overall*100, color="red", linestyle="--",
            label=f"Overall avg {overall:.1%}")
plt.title("Stockout Rate by Store")
plt.ylabel("Stockout Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_stockout_by_store.png"), dpi=150)
plt.show()

# Monthly trend
inv["month"] = inv["date"].dt.to_period("M")
monthly = inv.groupby("month")["stockout"].mean().reset_index()
monthly["m"] = monthly["month"].astype(str)

plt.figure(figsize=(14, 4))
plt.plot(monthly["m"], monthly["stockout"]*100, marker="o", color="#4C72B0")
plt.xticks(rotation=45, ha="right")
plt.title("Monthly Stockout Rate Over Time")
plt.ylabel("Stockout Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_stockout_monthly.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Class Imbalance
# %%
counts = feats["stockout"].value_counts()
labels = ["No Stockout (0)", "Stockout (1)"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].pie(counts.values, labels=labels, autopct="%1.1f%%",
            colors=["#4C72B0","#DD8452"], startangle=90)
axes[0].set_title("Class Distribution")

axes[1].bar(labels, counts.values, color=["#4C72B0","#DD8452"])
axes[1].set_yscale("log")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title("Class Counts")
for i, v in enumerate(counts.values):
    axes[1].text(i, v*1.2, f"{v:,}", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_class_imbalance.png"), dpi=150)
plt.show()
print(f"Imbalance ratio: {counts[0]//counts[1]}:1 (neg:pos)")

# %% [markdown]
# ## Feature Correlation Heatmap
# %%
num_cols = [
    "sales_lag_7","sales_lag_14","sales_lag_28",
    "sales_rolling_mean_7","sales_rolling_mean_28",
    "sales_velocity",
    "day_of_week","is_weekend","is_payday","is_holiday","is_snap",
    "days_to_festival","is_near_festival",
    "stock_ratio","consecutive_low_stock","stockout",
]
available = [c for c in num_cols if c in feats.columns]
corr = feats[available].corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, annot_kws={"size": 7})
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(MODELS, "eda_correlation.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Post-Training Evaluation
# %%
from IPython.display import Image, display

for fname in ["confusion_matrix.png", "feature_importance.png",
              "eda_class_imbalance.png"]:
    path = os.path.join(MODELS, fname)
    if os.path.exists(path):
        print(f"\n── {fname} ──")
        display(Image(path))
    else:
        print(f"{fname} not found — run train_model.py first")

report = os.path.join(MODELS, "evaluation_report.txt")
if os.path.exists(report):
    with open(report) as f:
        print(f.read())
else:
    print("No evaluation report yet — run scripts/train_model.py")
