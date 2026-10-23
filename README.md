# Decision Tree Classification Studio

A modular Streamlit app for uploading a CSV, exploring and preprocessing its columns, training custom ID3 and C4.5 classification trees, evaluating predictions, and trying single-row predictions.

## Run locally

Requires Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The included `data_breast_cancer.csv` opens as a starter dataset. Choose `Status` as its target to predict Alive/Dead. Upload other CSV files from the sidebar (maximum 50 MB).

## Features

- CSV preview, dimensions, dtypes, missing-value counts, and class-count/task detection.
- Mean/median numeric imputation; mode/constant categorical imputation; label or one-hot encoding.
- Reproducible train/test split, stratified when class counts permit.
- Custom recursive ID3 (entropy/information gain, categorical multiway splits) and C4.5 (gain ratio, categorical multiway and numeric threshold splits).
- Depth, minimum split size, and minimum information-gain controls.
- Graphviz tree diagrams, macro/weighted precision/recall/F1, accuracy, confusion matrices, comparative metrics and delta chart.
- Dynamic one-row prediction form for both trained models.

## Architecture

```text
CSV upload ──> app.py (Streamlit UI and workflow)
                  │
                  ├── src/preprocessing.py (cleaning, imputation, encoding, split)
                  ├── src/models.py (custom ID3 and C4.5 recursive trees, DOT output)
                  └── src/evaluation.py (metrics and confusion matrix)
```

## Algorithm notes

For a node with class entropy `H(T)`, information gain is `IG(T,a) = H(T) - Σᵥ (|Tᵥ|/|T|) H(Tᵥ)`. Split information is `SplitInfo(T,a) = -Σᵥ pᵥ log₂(pᵥ)` and gain ratio is `IG / SplitInfo`.

ID3 treats numeric values as categorical values. This is faithful to the classic discrete ID3 rule but may produce broad trees on continuous or high-cardinality data. C4.5 checks numeric midpoints and applies gain ratio; its candidate list is capped at 128 evenly spaced thresholds for high-cardinality features. This is an educational implementation rather than Quinlan's complete C4.5 implementation (which includes pruning, missing-value routing, and further safeguards). Neither algorithm here is CART; scikit-learn's `DecisionTreeClassifier` implements CART-style binary trees.

Preprocessing currently estimates imputations before splitting, which can introduce mild information leakage. For formal experiments, split first and fit preprocessing statistics on training data only. Performance results depend on target, split, and settings; the app displays live results rather than claiming a fixed benchmark.

## Source layout

```text
app.py
src/
  preprocessing.py
  models.py
  evaluation.py
data_breast_cancer.csv
requirements.txt
```

## GitHub

Initialize and maintain this directory as a Git repository, commit the source and documentation, then create/push to a GitHub repository using your GitHub account. No remote repository was supplied, so this workspace is not pushed to GitHub.
