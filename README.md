# Stock Price vs Fundamentals Analysis

### Author
Maaz Chunawala

### Objective
To analyze how stock prices correlate with fundamental financial metrics such as Sales Growth, EBITDA Margin Change, EBITDA Growth, PAT Growth, and PAT Margin Change.

---

## Steps Conducted
1. Loaded stock price and fundamentals data from Excel.
2. Cleaned, standardized, and merged both datasets.
3. Computed correlation matrix to find linear relationships.
4. Ran company-wise linear regressions (OLS method).
5. Summarized results with R² and top 3 significant variables.
6. Visualized correlation heatmap and regression results.

---

## Pre-Tests Conducted
- Checked for missing values.
- Ensured correct data types.
- Excluded companies with insufficient samples (<3 entries).

---

## Libraries Used
- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn  
- statsmodels  

---

## Output Files
| File | Description |
|------|--------------|
| `outputs/correlation_matrix.png` | Heatmap of variable correlations |
| `outputs/regression_summary.csv` | Regression summary per company |
| `outputs/r_squared_barplot.png` | R² score plot |
| `outputs/feature_importance.png` | Random Forest feature importance |

---

## Bonus Analyses
- Random Forest Feature Importance
- Possibility of multi-variate regression combining all companies
- PCA to identify principal drivers of stock price movement
- Time-series trend correlation (if data permits)

---

## Execution
Run the following in terminal:

```bash
source venv/bin/activate
python main.py