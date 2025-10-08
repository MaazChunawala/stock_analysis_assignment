import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
#  CONFIGURATION
# -----------------------------
DATA_PATH = "data/Intern test 2 - correlation regression - Copy.xls"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum years required per regression
MIN_YEARS_FOR_REGRESSION = 2
RANDOM_SEED = 42
N_RF_ESTIMATORS = 200

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------
#  HELPER FUNCTIONS
# -----------------------------
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def canonical_company(s):
    return re.sub(r"\s+", " ", safe_str(s)).strip().lower()

def extract_year_from_label(label):
    label = safe_str(label)
    m = re.search(r"(19|20)\d{2}", label)
    return int(m.group(0)) if m else None

def map_field_to_canonical(field):
    f = safe_str(field).lower()
    if "sale" in f:
        return "sales"
    if "ebitda" in f and "margin" in f:
        return "ebitda_margin"
    if "ebitda" in f:
        return "ebitda"
    if ("pat" in f) or ("net profit" in f) or ("profit after tax" in f):
        if "margin" in f:
            return "pat_margin"
        return "pat"
    if "margin" in f:
        return "margin"
    return None

# -----------------------------
#  1. LOAD DATA
# -----------------------------
logging.info(f"Reading Excel file: {DATA_PATH}")
xls = pd.ExcelFile(DATA_PATH)
logging.info(f"Available sheets: {xls.sheet_names}")

stock_raw = pd.read_excel(xls, sheet_name=0, header=None)
fund_raw = pd.read_excel(xls, sheet_name=1, header=0)

# -----------------------------
#  2. PARSE STOCK SHEET
# -----------------------------
name_row_idx = None
for i in range(min(12, stock_raw.shape[0])):
    v = safe_str(stock_raw.iat[i, 0]).strip().lower()
    if "name" in v:
        name_row_idx = i
        break

if name_row_idx is None:
    raise RuntimeError("Could not find 'Name' row in stock sheet.")

company_names = [canonical_company(x) for x in stock_raw.iloc[name_row_idx, 1:].tolist()]

rows = []
for r in range(name_row_idx + 1, stock_raw.shape[0]):
    label = safe_str(stock_raw.iat[r, 0]).strip()
    year = extract_year_from_label(label)
    if year is not None:
        rows.append((r, int(year), label))
    elif "current market value" in label.lower():
        rows.append((r, None, label))

stock_long = []
for r_idx, year, label in rows:
    for col_idx, comp in enumerate(company_names, start=1):
        val = pd.to_numeric(stock_raw.iat[r_idx, col_idx], errors="coerce")
        if np.isnan(val):
            continue
        stock_long.append({
            "CompanyName": comp,
            "Year": year,
            "Stock_Price": float(val),
            "Label": label
        })

stock_prices = pd.DataFrame(stock_long)
stock_prices["CompanyName"] = stock_prices["CompanyName"].astype(str)

# -----------------------------
#  3. PARSE FUNDAMENTALS SHEET
# -----------------------------
fund_raw.columns = [safe_str(c).strip() for c in fund_raw.columns]
company_col = next((c for c in fund_raw.columns if "company" in c.lower()), None)
field_col = next((c for c in fund_raw.columns if "field" in c.lower()), None)
year_cols = [c for c in fund_raw.columns if re.match(r"^(19|20)\d{2}$", str(c))]

fund_melt = fund_raw.melt(
    id_vars=[company_col, field_col],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value"
)
fund_melt["Year"] = pd.to_numeric(fund_melt["Year"], errors="coerce").astype("Int64")
fund_melt["Value"] = pd.to_numeric(fund_melt["Value"], errors="coerce")
fund_melt["CompanyName"] = fund_melt[company_col].astype(str).apply(canonical_company)
fund_melt["Field_canonical"] = fund_melt[field_col].apply(map_field_to_canonical)

# Drop rows without valid field mapping
fund_melt = fund_melt.dropna(subset=["Field_canonical", "Value"])

# Fill missing intermediate years per company-field pair
filled_records = []
for (comp, field), grp in fund_melt.groupby(["CompanyName", "Field_canonical"]):
    grp = grp.sort_values("Year")
    grp["Value"] = grp["Value"].interpolate(method="linear", limit_direction="both")
    grp["Value"] = grp["Value"].ffill().bfill()
    filled_records.append(grp)
fund_filled = pd.concat(filled_records, ignore_index=True)

# -----------------------------
#  4. COMPUTE GROWTH FEATURES
# -----------------------------
records = []
for (company, field), grp in fund_filled.groupby(["CompanyName", "Field_canonical"]):
    if grp["Value"].nunique() <= 1:
        continue
    grp = grp.sort_values("Year").set_index("Year")
    pct = grp["Value"].pct_change()
    for year, val in pct.items():
        if pd.notna(val):
            records.append({"CompanyName": company, "Year": year, "Field": field, "YoY": val})

features = pd.DataFrame(records)
feat_pivot = features.pivot_table(index=["CompanyName", "Year"], columns="Field", values="YoY").reset_index()

# Add missing fields
for col in ["sales", "ebitda", "pat", "ebitda_margin", "pat_margin"]:
    if col not in feat_pivot.columns:
        feat_pivot[col] = np.nan

feat_pivot.rename(columns={
    "sales": "Sales_Growth",
    "ebitda": "EBITDA_Growth",
    "pat": "PAT_Growth",
    "ebitda_margin": "EBITDA_Margin_Change",
    "pat_margin": "PAT_Margin_Change"
}, inplace=True)

# -----------------------------
#  5. ALIGN STOCK & FUNDAMENTALS
# -----------------------------
stock_prices["Year"] = stock_prices["Year"].fillna(
    stock_prices.groupby("CompanyName")["Year"].transform("max")
).astype("Int64")

data = pd.merge(
    stock_prices,
    feat_pivot,
    on=["CompanyName", "Year"],
    how="left"
)

feature_cols = [
    "Sales_Growth",
    "EBITDA_Growth",
    "EBITDA_Margin_Change",
    "PAT_Growth",
    "PAT_Margin_Change"
]

# -----------------------------
#  6. PER-COMPANY ANALYSIS
# -----------------------------
regression_results = []

for comp, dfc in data.groupby("CompanyName"):
    dfc = dfc.dropna(subset=["Stock_Price"])
    if dfc.shape[0] < MIN_YEARS_FOR_REGRESSION:
        logging.info(f"Skipping {comp}: only {dfc.shape[0]} data points")
        continue

    corr = dfc[["Stock_Price"] + feature_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Correlation: {comp}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"corr_{re.sub(r'[^a-z0-9]', '_', comp)}.png"), dpi=150)
    plt.close()

    X = dfc[feature_cols].fillna(0)
    y = dfc["Stock_Price"].ffill().bfill()
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()

    coeffs = model.params.to_dict()
    pvals = model.pvalues.to_dict()
    r2 = float(model.rsquared)

    top3 = sorted([(k, v) for k, v in pvals.items() if k != "const"], key=lambda x: x[1])[:3]
    top3_vars = [k for k, _ in top3]

    regression_results.append({
        "CompanyName": comp,
        "R_squared": r2,
        "Top3_Significant_Variables": ", ".join(top3_vars),
        "Coefficients": "; ".join(f"{k}: {v:.4f}" for k, v in coeffs.items()),
        "P_values": "; ".join(f"{k}: {v:.4f}" for k, v in pvals.items())
    })

reg_summary = pd.DataFrame(regression_results)
reg_summary.to_csv(os.path.join(OUTPUT_DIR, "regression_summary.csv"), index=False)

# -----------------------------
#  7. GLOBAL FEATURE IMPORTANCE
# -----------------------------
global_df = data.dropna(subset=["Stock_Price"])
if global_df.shape[0] >= 2:
    Xg = global_df[feature_cols].fillna(0)
    yg = global_df["Stock_Price"]
    rf = RandomForestRegressor(n_estimators=N_RF_ESTIMATORS, random_state=RANDOM_SEED)
    rf.fit(Xg, yg)
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(6, 4))
    imp.plot(kind="barh")
    plt.title("Global Feature Importance (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "global_feature_importance.png"), dpi=150)
    plt.close()

    imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))

logging.info("✅ All outputs saved in 'outputs/' folder successfully.")