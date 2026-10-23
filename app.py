from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.preprocessing import clean_frame, preprocess, split_data
from src.models import DecisionTree
from src.evaluation import evaluate

st.set_page_config(page_title="Decision Tree Studio", page_icon="🌳", layout="wide")
st.title("🌳 Decision Tree Classification Studio")
st.caption("Explore your data, train Information Gain and Gain Ratio trees, and compare their predictions.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a classification dataset", type=["csv"], help="CSV files up to 50 MB")
    if uploaded and uploaded.size > 50 * 1024 * 1024:
        st.error("The file exceeds the 50 MB upload limit.")
        st.stop()
    if uploaded:
        try:
            raw = clean_frame(pd.read_csv(uploaded))
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()
    else:
        st.info("Upload a CSV to begin, or explore the included breast cancer dataset.")
        try:
            raw = clean_frame(pd.read_csv("data_breast_cancer.csv"))
        except Exception:
            st.stop()
    target = st.selectbox("Target column", list(raw.columns), index=list(raw.columns).index("Status") if "Status" in raw.columns else len(raw.columns)-1)
    st.divider()
    st.header("Preprocessing")
    numeric_imputation = st.selectbox("Numeric missing values", ["median", "mean"])
    categorical_imputation = st.selectbox("Categorical missing values", ["mode", "constant"])
    constant = st.text_input("Constant for categorical values", "Unknown")
    encoding = st.selectbox("Categorical encoding", ["Label Encoding", "One-Hot Encoding"])
    test_size = st.slider("Test set (%)", 10, 40, 20, 5) / 100
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    st.divider()
    st.header("Tree settings")
    max_depth = st.slider("Maximum depth", 1, 20, 6)
    min_samples_split = st.slider("Minimum samples to split", 2, 50, 4)
    min_gain = st.number_input("Minimum information gain", 0.0, 1.0, 0.0, 0.001, format="%.3f")

st.subheader("Dataset overview")
meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Rows", f"{len(raw):,}")
meta2.metric("Columns", len(raw.columns))
meta3.metric("Target classes", raw[target].nunique(dropna=True))
meta4.metric("Task", "Binary" if raw[target].nunique(dropna=True) == 2 else "Multi-class")
with st.expander("Explore data", expanded=True):
    left, right = st.columns(2)
    with left:
        view = st.radio("Preview", ["Head", "Tail"], horizontal=True)
        st.dataframe(raw.head(10) if view == "Head" else raw.tail(10), use_container_width=True)
    with right:
        st.markdown("**Column types**")
        st.dataframe(pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                                   "Missing": raw.isna().sum().values,
                                   "Unique": raw.nunique(dropna=True).values}), hide_index=True, use_container_width=True)

try:
    X, y, feature_kinds = preprocess(raw, target, numeric_imputation, categorical_imputation, constant, encoding)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, int(seed))
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

if y.nunique() < 2:
    st.error("The selected target must contain at least two classes.")
    st.stop()
if y_train.nunique() < 2:
    st.error("The training split contains only one target class. Adjust the test size or choose another seed.")
    st.stop()

with st.expander("Train/test details"):
    st.write(f"Training rows: **{len(X_train):,}** · Test rows: **{len(X_test):,}** · Features after encoding: **{X.shape[1]}**")
    st.write("Stratification is used when every target class has at least two examples.")

models, metrics, predictions = {}, {}, {}
for algorithm in ["ID3", "C4.5"]:
    model = DecisionTree(algorithm, feature_kinds, max_depth, min_samples_split, min_gain).fit(X_train, y_train)
    pred = model.predict(X_test)
    models[algorithm] = model
    predictions[algorithm] = pred
    metrics[algorithm] = evaluate(y_test, pred, sorted(y.unique()))

st.subheader("Model comparison")
metric_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
comparison = pd.DataFrame({name: {k: v[k] for k in metric_names} for name, v in metrics.items()}).T
st.dataframe(comparison.style.format("{:.3f}").highlight_max(axis=0, color="#d1fae5"), use_container_width=True)
fig, ax = plt.subplots(figsize=(9, 3.4))
delta = comparison.loc["C4.5"] - comparison.loc["ID3"]
colors = ["#16a085" if v >= 0 else "#e76f51" for v in delta]
ax.bar(delta.index, delta.values, color=colors)
ax.axhline(0, color="#64748b", linewidth=0.8)
ax.set_ylabel("C4.5 minus ID3")
ax.set_title("Metric difference (positive favors C4.5)")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

tabs = st.tabs(["Decision trees", "Confusion matrices", "Prediction sandbox", "Algorithm notes"])
with tabs[0]:
    algo = st.radio("Tree", ["ID3", "C4.5"], horizontal=True, key="tree_choice")
    st.graphviz_chart(models[algo].to_dot(), use_container_width=True)
    st.caption(f"{algo} · test accuracy {metrics[algo]['Accuracy']:.3f}. Tree labels include entropy, information gain, sample count, and majority class.")
with tabs[1]:
    cols = st.columns(2)
    labels = sorted(y.unique())
    for col, algo in zip(cols, ["ID3", "C4.5"]):
        with col:
            st.markdown(f"**{algo}**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(metrics[algo]["Confusion matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
with tabs[2]:
    st.write("Enter values in the model's encoded feature space. For label encoding, categorical values are shown as integer codes.")
    with st.form("prediction_form"):
        inputs = {}
        input_cols = st.columns(2)
        for i, feature in enumerate(X.columns):
            col = input_cols[i % 2]
            if feature_kinds.get(feature) == "categorical":
                options = sorted(pd.Series(X[feature].dropna().unique()).tolist())
                inputs[feature] = col.selectbox(feature, options, format_func=lambda v: f"{v} (code)")
            else:
                inputs[feature] = col.number_input(feature, value=float(X[feature].median()), key=f"predict_{feature}")
        submitted = st.form_submit_button("Predict with both models", type="primary")
    if submitted:
        sample = pd.DataFrame([inputs], columns=X.columns)
        for algo in ["ID3", "C4.5"]:
            result = models[algo].predict(sample)[0]
            st.success(f"{algo} prediction: **{result}**")
with tabs[3]:
    st.markdown("""
**ID3** uses Shannon entropy and information gain. It treats every feature as categorical and creates one branch per value.

**C4.5** selects by gain ratio (information gain divided by split information). Numeric features are evaluated at candidate thresholds; categorical features use multiway splits. Gain ratio reduces the preference for attributes with many distinct values.

These are lightweight teaching implementations, not scikit-learn CART. ID3 does not support continuous-threshold splits. C4.5 threshold search is capped at 128 evenly spaced candidate positions on very high-cardinality columns to keep interactive training responsive. Missing values are imputed before the split; for rigorous benchmark work, fit imputation statistics on training data only.
""")
