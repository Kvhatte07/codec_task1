"""
Customer Churn Prediction Pipeline
====================================
Telecom dataset · EDA + ML models + Evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import json

# ─────────────────────────────────────────────
# PALETTE (inspired by signal / risk telemetry)
# ─────────────────────────────────────────────
BG      = "#0D1117"
PANEL   = "#161B22"
BORDER  = "#21262D"
ACCENT  = "#58A6FF"   # cool blue – primary accent
WARN    = "#F78166"   # coral red  – churn / risk
SUCCESS = "#3FB950"   # green      – retained
MUTED   = "#8B949E"   # grey       – secondary text
WHITE   = "#E6EDF3"
GOLD    = "#D29922"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": WHITE,
    "text.color": WHITE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": BORDER,
    "grid.linewidth": 0.6,
    "font.family": "monospace",
    "legend.facecolor": PANEL,
    "legend.edgecolor": BORDER,
})

# ─────────────────────────────────────────────
# 1. SYNTHETIC TELECOM DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 7_000

def make_dataset(n):
    tenure       = np.random.exponential(30, n).clip(1, 72).astype(int)
    monthly      = np.random.normal(65, 30, n).clip(18, 120)
    num_products = np.random.choice([1,2,3,4], n, p=[0.3,0.35,0.25,0.1])
    support_calls = np.random.poisson(1.5, n).clip(0, 10)
    contract     = np.random.choice(["Month-to-Month","One Year","Two Year"],
                                     n, p=[0.55, 0.25, 0.20])
    internet     = np.random.choice(["Fiber","DSL","No"], n, p=[0.45,0.35,0.20])
    payment      = np.random.choice(["Electronic","Mailed","Bank Transfer","Credit Card"],
                                     n, p=[0.35, 0.20, 0.25, 0.20])
    senior       = np.random.choice([0, 1], n, p=[0.84, 0.16])
    dependents   = np.random.choice([0, 1], n, p=[0.60, 0.40])

    # Churn probability — correlated with real-world drivers
    log_odds = (
        -1.8
        - 0.055 * tenure
        + 0.018 * monthly
        - 0.30  * num_products
        + 0.35  * support_calls
        + 1.20  * (contract == "Month-to-Month").astype(int)
        - 0.50  * (contract == "Two Year").astype(int)
        + 0.40  * (internet == "Fiber").astype(int)
        + 0.30  * (payment == "Electronic").astype(int)
        + 0.25  * senior
        - 0.20  * dependents
        + np.random.normal(0, 0.4, n)
    )
    prob  = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly.round(2),
        "total_charges": (monthly * tenure * np.random.uniform(0.85, 1.05, n)).round(2),
        "num_products": num_products,
        "support_calls": support_calls,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "dependents": dependents,
        "churn": churn,
    })

df = make_dataset(N)
print(f"Dataset: {df.shape[0]:,} rows · {df.shape[1]} columns")
print(f"Churn rate: {df['churn'].mean()*100:.1f}%\n")

# ─────────────────────────────────────────────
# 2. EDA  →  Figure 1: "Signal Dashboard"
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle("CUSTOMER CHURN  ·  SIGNAL DASHBOARD",
             fontsize=17, fontweight="bold", color=WHITE,
             x=0.5, y=0.98)
fig.text(0.5, 0.955, "Exploratory data analysis · telecom cohort (7,000 customers)",
         ha="center", fontsize=9, color=MUTED)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45,
                       top=0.93, bottom=0.06, left=0.06, right=0.97)

churned  = df[df["churn"] == 1]
retained = df[df["churn"] == 0]

# ── 2a. Churn donut ─────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
sizes  = [df["churn"].sum(), (df["churn"] == 0).sum()]
colors = [WARN, SUCCESS]
wedges, _ = ax0.pie(sizes, colors=colors, startangle=90,
                    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
ax0.text(0, 0, f"{df['churn'].mean()*100:.1f}%\nCHURN", ha="center", va="center",
         fontsize=13, fontweight="bold", color=WARN)
ax0.set_title("Churn Rate", color=WHITE, fontsize=10, pad=8)
patches = [mpatches.Patch(color=WARN, label="Churned"),
           mpatches.Patch(color=SUCCESS, label="Retained")]
ax0.legend(handles=patches, loc="lower center", fontsize=7.5,
           bbox_to_anchor=(0.5, -0.18), ncol=2)

# ── 2b. Tenure distribution ─────────────────
ax1 = fig.add_subplot(gs[0, 1:3])
bins = np.linspace(0, 72, 37)
ax1.hist(retained["tenure"], bins=bins, color=SUCCESS, alpha=0.6, label="Retained")
ax1.hist(churned["tenure"],  bins=bins, color=WARN,    alpha=0.7, label="Churned")
ax1.set_title("Tenure (months)", color=WHITE, fontsize=10)
ax1.set_xlabel("Months", fontsize=8); ax1.set_ylabel("Count", fontsize=8)
ax1.legend(fontsize=8); ax1.grid(axis="y")

# ── 2c. Monthly charges KDE ─────────────────
ax2 = fig.add_subplot(gs[0, 3])
for subset, color, lbl in [(retained, SUCCESS, "Retained"),
                            (churned,  WARN,    "Churned")]:
    vals = subset["monthly_charges"].values
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    from scipy.ndimage import gaussian_filter1d
    hist, edges = np.histogram(vals, bins=60, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth  = gaussian_filter1d(hist, sigma=2)
    ax2.plot(centers, smooth, color=color, lw=2, label=lbl)
    ax2.fill_between(centers, smooth, alpha=0.15, color=color)
ax2.set_title("Monthly Charges", color=WHITE, fontsize=10)
ax2.set_xlabel("USD", fontsize=8); ax2.set_ylabel("Density", fontsize=8)
ax2.legend(fontsize=8)

# ── 2d. Contract type churn rate ─────────────
ax3 = fig.add_subplot(gs[1, 0:2])
contract_churn = (df.groupby("contract")["churn"]
                    .agg(["mean","count"])
                    .rename(columns={"mean":"rate","count":"n"}))
bars = ax3.bar(contract_churn.index, contract_churn["rate"]*100,
               color=[WARN, ACCENT, SUCCESS], width=0.5, zorder=3)
ax3.set_title("Churn Rate by Contract Type", color=WHITE, fontsize=10)
ax3.set_ylabel("Churn %", fontsize=8); ax3.grid(axis="y", zorder=0)
ax3.set_ylim(0, 60)
for bar, (_, row) in zip(bars, contract_churn.iterrows()):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1.2,
             f"{row['rate']*100:.1f}%\n(n={int(row['n']):,})",
             ha="center", fontsize=7.5, color=WHITE)

# ── 2e. Support calls vs churn ───────────────
ax4 = fig.add_subplot(gs[1, 2:4])
calls_churn = df.groupby("support_calls")["churn"].mean() * 100
ax4.plot(calls_churn.index, calls_churn.values,
         color=WARN, lw=2.5, marker="o", markersize=6, zorder=3)
ax4.fill_between(calls_churn.index, calls_churn.values,
                 alpha=0.15, color=WARN)
ax4.set_title("Support Calls → Churn Probability", color=WHITE, fontsize=10)
ax4.set_xlabel("Number of support calls", fontsize=8)
ax4.set_ylabel("Churn rate %", fontsize=8); ax4.grid(axis="y", zorder=0)

# ── 2f. Correlation heatmap ──────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
num_cols = ["tenure","monthly_charges","total_charges",
            "num_products","support_calls","senior_citizen","dependents","churn"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax5,
            linewidths=0.5, linecolor=BG,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.7})
ax5.set_title("Feature Correlation", color=WHITE, fontsize=10)
ax5.tick_params(axis="x", rotation=45, labelsize=7)
ax5.tick_params(axis="y", rotation=0,  labelsize=7)

# ── 2g. Internet service churn ───────────────
ax6 = fig.add_subplot(gs[2, 2:4])
inet_churn = (df.groupby("internet_service")["churn"]
                .mean().sort_values(ascending=False) * 100)
colors_inet = [WARN if v > 25 else ACCENT for v in inet_churn.values]
bars2 = ax6.barh(inet_churn.index, inet_churn.values,
                 color=colors_inet, height=0.5, zorder=3)
ax6.set_title("Churn Rate by Internet Service", color=WHITE, fontsize=10)
ax6.set_xlabel("Churn %", fontsize=8); ax6.grid(axis="x", zorder=0)
for bar in bars2:
    ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.1f}%", va="center", fontsize=8, color=WHITE)

plt.savefig("/home/claude/churn_prediction/outputs/fig1_eda_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  Fig 1: EDA dashboard saved")

# ─────────────────────────────────────────────
# 3. PRE-PROCESSING
# ─────────────────────────────────────────────
df_ml = df.copy()
le = LabelEncoder()
for col in ["contract", "internet_service", "payment_method"]:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml.drop("churn", axis=1)
y = df_ml["churn"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=10,
                                                   min_samples_leaf=5, class_weight="balanced",
                                                   random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                          subsample=0.8, colsample_bytree=0.8,
                                          scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                          use_label_encoder=False, eval_metric="logloss",
                                          random_state=42, n_jobs=-1),
}

results   = {}
roc_data  = {}
proba_all = {}

for name, mdl in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test

    mdl.fit(Xtr, y_train)
    y_pred  = mdl.predict(Xte)
    y_proba = mdl.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = dict(accuracy=acc, recall=rec, f1=f1, roc_auc=auc, model=mdl)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name]  = (fpr, tpr, auc)
    proba_all[name] = y_proba

    print(f"  {name:<24} Accuracy={acc:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# ─────────────────────────────────────────────
# 5. EVALUATION  →  Figure 2
# ─────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": ACCENT,
    "Random Forest":       SUCCESS,
    "XGBoost":             GOLD,
}

fig2, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=BG)
fig2.suptitle("MODEL EVALUATION  ·  CUSTOMER CHURN PREDICTION",
              fontsize=15, fontweight="bold", color=WHITE, y=0.97)
fig2.text(0.5, 0.935, "Logistic Regression · Random Forest · XGBoost  |  test set n=1,400",
          ha="center", fontsize=9, color=MUTED)

# ── 5a. Grouped metric bars ───────────────────
ax = axes[0, 0]
metrics   = ["accuracy", "recall", "f1", "roc_auc"]
x         = np.arange(len(metrics))
width     = 0.22
labels_m  = ["Accuracy", "Recall", "F1", "ROC-AUC"]

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * width - width, vals, width,
                  label=name, color=MODEL_COLORS[name], alpha=0.88, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=6.5, color=WHITE)

ax.set_xticks(x); ax.set_xticklabels(labels_m, fontsize=9)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score", fontsize=9)
ax.set_title("Performance Metrics", color=WHITE, fontsize=11)
ax.legend(fontsize=8); ax.grid(axis="y", zorder=0)

# ── 5b. ROC curves ───────────────────────────
ax = axes[0, 1]
ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="Random (0.50)")
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=2.5, color=MODEL_COLORS[name],
            label=f"{name} ({auc:.3f})")
ax.fill_between(roc_data["XGBoost"][0], roc_data["XGBoost"][1],
                alpha=0.08, color=GOLD)
ax.set_xlabel("False Positive Rate", fontsize=9)
ax.set_ylabel("True Positive Rate", fontsize=9)
ax.set_title("ROC Curves", color=WHITE, fontsize=11)
ax.legend(fontsize=8); ax.grid()

# ── 5c. Confusion matrix – XGBoost ───────────
ax = axes[0, 2]
best_name = max(results, key=lambda k: results[k]["roc_auc"])
Xte_best  = X_test_sc if best_name == "Logistic Regression" else X_test
cm = confusion_matrix(y_test, results[best_name]["model"].predict(Xte_best))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=1, linecolor=BG,
            xticklabels=["Retained","Churned"],
            yticklabels=["Retained","Churned"],
            annot_kws={"size": 13})
ax.set_title(f"Confusion Matrix – {best_name}", color=WHITE, fontsize=11)
ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)

# ── 5d. Feature importance – XGBoost ─────────
ax = axes[1, 0]
xgb_mdl = results["XGBoost"]["model"]
importances = pd.Series(xgb_mdl.feature_importances_, index=feature_names).sort_values()
colors_fi = [WARN if importances[f] > importances.median() else ACCENT
             for f in importances.index]
ax.barh(importances.index, importances.values, color=colors_fi, height=0.6, zorder=3)
ax.set_title("Feature Importance – XGBoost", color=WHITE, fontsize=11)
ax.set_xlabel("Importance score", fontsize=9); ax.grid(axis="x", zorder=0)

# ── 5e. Probability distribution ─────────────
ax = axes[1, 1]
from scipy.ndimage import gaussian_filter1d
for name, proba in proba_all.items():
    hist_c, edges = np.histogram(proba[y_test==1], bins=50, density=True)
    centers = (edges[:-1]+edges[1:])/2
    ax.plot(centers, gaussian_filter1d(hist_c, 2),
            color=MODEL_COLORS[name], lw=2, label=name)
ax.set_title("Predicted Churn Probability\n(actual churners)", color=WHITE, fontsize=10)
ax.set_xlabel("P(churn)", fontsize=9); ax.set_ylabel("Density", fontsize=9)
ax.legend(fontsize=8); ax.grid()

# ── 5f. Cross-val scores ──────────────────────
ax = axes[1, 2]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, mdl in models.items():
    Xtr_cv = X_train_sc if name == "Logistic Regression" else X_train
    scores = cross_val_score(mdl, Xtr_cv, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores

positions = list(range(len(cv_results)))
for pos, (name, scores) in zip(positions, cv_results.items()):
    ax.boxplot(scores, positions=[pos], widths=0.4,
               patch_artist=True,
               boxprops=dict(facecolor=MODEL_COLORS[name], alpha=0.7, color=WHITE),
               whiskerprops=dict(color=WHITE, lw=1.5),
               capprops=dict(color=WHITE, lw=1.5),
               medianprops=dict(color=WHITE, lw=2.5),
               flierprops=dict(marker="o", markerfacecolor=WARN, markersize=5))
    ax.text(pos, scores.mean() - 0.018, f"{scores.mean():.3f}",
            ha="center", fontsize=8.5, color=WHITE, fontweight="bold")

ax.set_xticks(positions)
ax.set_xticklabels([n.replace(" ","\n") for n in cv_results], fontsize=8)
ax.set_ylabel("ROC-AUC (5-fold CV)", fontsize=9)
ax.set_title("Cross-Validation Stability", color=WHITE, fontsize=11)
ax.set_ylim(0.78, 0.96); ax.grid(axis="y")

plt.savefig("/home/claude/churn_prediction/outputs/fig2_model_evaluation.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  Fig 2: Model evaluation saved")

# ─────────────────────────────────────────────
# 6. BUSINESS RISK TIERS  →  Figure 3
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig3.suptitle("CHURN RISK TIERS  ·  BUSINESS INTELLIGENCE VIEW",
              fontsize=14, fontweight="bold", color=WHITE, y=1.01)

xgb_proba_test = proba_all["XGBoost"]
risk_df = X_test.copy()
risk_df["churn_prob"] = xgb_proba_test
risk_df["actual"]     = y_test.values
risk_df["risk_tier"]  = pd.cut(xgb_proba_test,
                                 bins=[0, 0.30, 0.60, 1.0],
                                 labels=["Low Risk", "Medium Risk", "High Risk"])

# 6a. Tier distribution
ax = axes3[0]
tier_counts = risk_df["risk_tier"].value_counts().reindex(["Low Risk","Medium Risk","High Risk"])
tier_colors = [SUCCESS, GOLD, WARN]
bars = ax.bar(tier_counts.index, tier_counts.values,
              color=tier_colors, width=0.55, zorder=3)
ax.set_title("Customer Risk Distribution", color=WHITE, fontsize=11)
ax.set_ylabel("Count", fontsize=9); ax.grid(axis="y", zorder=0)
for bar, n in zip(bars, tier_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{n:,}\n({n/len(risk_df)*100:.0f}%)",
            ha="center", fontsize=8.5, color=WHITE)

# 6b. Actual churn rate per tier
ax = axes3[1]
tier_accuracy = risk_df.groupby("risk_tier")["actual"].mean() * 100
tier_accuracy = tier_accuracy.reindex(["Low Risk","Medium Risk","High Risk"])
bars2 = ax.bar(tier_accuracy.index, tier_accuracy.values,
               color=tier_colors, width=0.55, zorder=3)
ax.set_title("Actual Churn Rate per Tier", color=WHITE, fontsize=11)
ax.set_ylabel("Actual Churn %", fontsize=9); ax.grid(axis="y", zorder=0)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", fontsize=10,
            fontweight="bold", color=WHITE)

# 6c. Avg monthly charges per tier
ax = axes3[2]
tier_charges = risk_df.groupby("risk_tier")["monthly_charges"].mean()
tier_charges = tier_charges.reindex(["Low Risk","Medium Risk","High Risk"])
bars3 = ax.bar(tier_charges.index, tier_charges.values,
               color=tier_colors, width=0.55, zorder=3, alpha=0.85)
ax.set_title("Avg Monthly Revenue at Risk", color=WHITE, fontsize=11)
ax.set_ylabel("Avg Monthly Charges (USD)", fontsize=9); ax.grid(axis="y", zorder=0)
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"${bar.get_height():.0f}", ha="center", fontsize=10,
            fontweight="bold", color=WHITE)

plt.tight_layout()
plt.savefig("/home/claude/churn_prediction/outputs/fig3_risk_tiers.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  Fig 3: Risk tiers saved")

# ─────────────────────────────────────────────
# 7. SUMMARY JSON
# ─────────────────────────────────────────────
summary = {m: {k: round(v,4) for k,v in r.items() if k != "model"}
           for m, r in results.items()}
best = max(results, key=lambda k: results[k]["roc_auc"])
summary["best_model"] = best
summary["dataset"] = {"rows": N, "churn_rate": round(df["churn"].mean(), 4)}

with open("/home/claude/churn_prediction/outputs/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n── FINAL SUMMARY ──────────────────────────────────────")
for name, metrics_d in summary.items():
    if isinstance(metrics_d, dict):
        print(f"  {name}: {metrics_d}")
print(f"\n✓  Best model: {best}")
print("\nAll outputs saved to /home/claude/churn_prediction/outputs/")
