# PRML Projects — Decision Tree Studio

An interactive React dashboard for exploring classification trees and decision costs. Import a CSV, choose its target column, rebuild ID3 or C4.5 as parameters change, inspect the tree and holdout metrics, and compare confusion counts with realized loss. CSV data is processed locally in the browser.

## Run the React dashboard

Requires Node.js 20.19+ or 22.12+.

```bash
npm install
npm run dev
```

Use the local URL printed by Vite. Create a static production build with `npm run build` and serve `dist/` from any static web server.

## Dashboard capabilities

- Import arbitrary UTF-8 CSV datasets with headers and select any target column.
- Inspect dataset dimensions, missing values, class balance, and a row preview.
- Rebuild ID3/C4.5 trees interactively with depth, minimum leaf size, holdout split, and class-prior controls.
- Edit the action-by-class loss matrix and inspect conditional risk curves and the resulting decision threshold.
- Compare accuracy, macro precision/recall/F1, confusion counts, and accumulated holdout loss.
- Export the current tree and evaluation snapshot as JSON.

The browser implementation supports classification data with modest feature and class counts. It includes numeric midpoint splitting, categorical splits, deterministic stratified holdout, and train-fitted missing-value imputation. The binary risk plot applies to two-class targets; multiclass targets use the risk-vector display.

## Python classifier and Streamlit reference app

The Python classifier, preprocessing, and evaluation modules remain available for command-line and research workflows. The original Streamlit interface is retained in `app.py` for compatibility; the React dashboard is the primary interface.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run a classifier smoke check by importing `DecisionTree` from `src.models`. Details on algorithms and current Python preprocessing behavior are documented in the module docstrings and source.

## Repository layout

```text
index.html                 React entry document
src/main.jsx               React entry point
src/DecisionStudio.jsx     dashboard, tree builder, risk and metric views
src/technical.css          technical grid and color system
public/data_breast_cancer.csv  bundled starter dataset
app.py                     retained Streamlit reference application
src/models.py              Python ID3/C4.5 classifier
src/preprocessing.py       Python data preparation
src/evaluation.py          Python metric helpers
```
