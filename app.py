
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None

    def impute_missing(
        self, df, target_col,
        strategy_num="median",
        strategy_cat="mode"
    ):
        df = df.copy()
        df = df.dropna(subset=[target_col])

        for col in df.columns:
            if col == target_col:
                continue

            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if strategy_num == "median":
                        value = df[col].median()
                    else:
                        value = df[col].mean()
                else:
                    if strategy_cat == "constant":
                        value = "Unknown"
                    else:
                        mode_values = df[col].mode()
                        value = (
                            mode_values.iloc[0]
                            if not mode_values.empty
                            else "Unknown"
                        )

                df[col] = df[col].fillna(value)

        return df

    def encode_features(
        self, df, target_col,
        encoding_type="Label Encoding"
    ):
        df = df.copy()

        self.target_encoder = LabelEncoder()
        df[target_col] = self.target_encoder.fit_transform(
            df[target_col].astype(str)
        )

        feature_cols = [
            col for col in df.columns
            if col != target_col
        ]

        categorical_features = [
            col for col in feature_cols
            if not pd.api.types.is_numeric_dtype(df[col])
        ]

        if (
            encoding_type == "One-Hot Encoding"
            and categorical_features
        ):
            df = pd.get_dummies(
                df,
                columns=categorical_features,
                drop_first=False,
                dtype=int,
            )
        else:
            for col in categorical_features:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(
                    df[col].astype(str)
                )
                self.label_encoders[col] = encoder

        feature_cols = [
            col for col in df.columns
            if col != target_col
        ]

        return df, feature_cols

    def split_data(
        self, df, target_col, feature_cols,
        test_size, random_state
    ):
        X = df[feature_cols].to_numpy().astype(float)
        y = df[target_col].to_numpy().astype(int)

        _, counts = np.unique(y, return_counts=True)
        stratify = (
            y if len(counts) > 1 and counts.min() >= 2
            else None
        )

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=stratify,
        )


class DecisionTreeNode:
    def __init__(
        self, feature=None, threshold=None,
        branches=None, value=None,
        entropy=0.0, samples=0, distribution=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.branches = branches or {}
        self.value = value
        self.entropy = entropy
        self.samples = samples
        self.distribution = distribution or []

    def is_leaf(self):
        return self.value is not None

class DecisionTreeStudioEngine:
    def __init__(
        self, criterion="C4.5",
        max_depth=6, min_samples_split=2,
        feature_names=None
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_names = feature_names
        self.root = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return -np.sum(
            probabilities * np.log2(probabilities)
        )

    def _best_split(self, X, y, available_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n = len(y)
        parent_entropy = self._entropy(y)

        for feature in available_features:
            values = np.unique(X[:, feature])

            if len(values) <= 1:
                continue

            thresholds = (
                values[:-1] + values[1:]
            ) / 2.0

            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                n_left = np.sum(mask)
                n_right = np.sum(~mask)

                if n_left == 0 or n_right == 0:
                    continue

                information_gain = parent_entropy - (
                    n_left / n * self._entropy(y[mask])
                    + n_right / n * self._entropy(y[~mask])
                )

                gain = information_gain

                if self.criterion == "C4.5":
                    p_left = n_left / n
                    p_right = n_right / n

                    split_info = -(
                        p_left * np.log2(p_left)
                        + p_right * np.log2(p_right)
                    )

                    gain = (
                        information_gain / split_info
                        if split_info > 0 else 0
                    )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def fit(self, X, y):
        self.root = self._build(
            X, y, 0, list(range(X.shape[1]))
        )
        return self

    def _build(self, X, y, depth, features):
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        distribution = [
            int(np.sum(y == cls)) for cls in classes
        ]

        if (
            len(classes) <= 1
            or len(y) < self.min_samples_split
            or depth >= self.max_depth
            or not features
        ):
            return DecisionTreeNode(
                value=int(majority_class),
                entropy=self._entropy(y),
                samples=len(y),
                distribution=distribution,
            )

        feature, threshold, gain = self._best_split(
            X, y, features
        )

        if feature is None or gain <= 0:
            return DecisionTreeNode(
                value=int(majority_class),
                entropy=self._entropy(y),
                samples=len(y),
                distribution=distribution,
            )

        node = DecisionTreeNode(
            feature=feature,
            threshold=threshold,
            entropy=self._entropy(y),
            samples=len(y),
            distribution=distribution,
        )

        mask = X[:, feature] <= threshold

        node.branches["left"] = self._build(
            X[mask], y[mask], depth + 1, features
        )

        node.branches["right"] = self._build(
            X[~mask], y[~mask], depth + 1, features
        )

        return node

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            child = node.branches["left"]
        else:
            child = node.branches["right"]

        return self._predict_row(child, x)

    def predict(self, X):
        return np.array([
            self._predict_row(self.root, row)
            for row in X
        ])

    def to_dot(self):
        dot = [
            "digraph Tree {",
            'node [shape=box, style="filled, rounded", '
            'fontname="helvetica"];'
        ]

        def traverse(node, node_id=0):
            current_id = node_id

            if node.is_leaf():
                dot.append(
                    f'{current_id} [label="Leaf\\n'
                    f'Entropy: {node.entropy:.2f}\\n'
                    f'Samples: {node.samples}\\n'
                    f'Class: {node.value}", '
                    'fillcolor="#d1e7dd"];'
                )
                return current_id

            name = (
                self.feature_names[node.feature]
                if self.feature_names
                else f"X[{node.feature}]"
            )
            name = str(name).replace('"', '\\"')

            dot.append(
                f'{current_id} [label="{name} <= '
                f'{node.threshold:.2f}\\n'
                f'Entropy: {node.entropy:.2f}\\n'
                f'Samples: {node.samples}", '
                'fillcolor="#cfe2ff"];'
            )

            left_id = current_id + 1
            dot.append(
                f'{current_id} -> {left_id} [label="True"];'
            )

            current_id = traverse(
                node.branches["left"], left_id
            )

            right_id = current_id + 1
            dot.append(
                f'{node_id} -> {right_id} [label="False"];'
            )

            current_id = traverse(
                node.branches["right"], right_id
            )

            return current_id

        traverse(self.root)
        dot.append("}")
        return "\n".join(dot)

st.set_page_config(
    page_title="Tree Studio Canvas",
    layout="wide"
)

st.title("🌳 Interactive Decision Tree Classification Studio")

if "models" not in st.session_state:
    st.session_state["models"] = {}

if "data" not in st.session_state:
    st.session_state["data"] = None

with st.sidebar:
    st.header("⚙️ Configuration Controls")

    uploaded_file = st.file_uploader(
        "Upload Target Classification CSV",
        type=["csv"]
    )

    st.markdown("### Preprocessing Options")

    impute_num = st.selectbox(
        "Numerical Missing Values",
        ["median", "mean"]
    )

    impute_cat = st.selectbox(
        "Categorical Missing Values",
        ["mode", "constant"]
    )

    encoding_method = st.selectbox(
        "Categorical Encoding",
        ["Label Encoding", "One-Hot Encoding"]
    )

    st.markdown("### Dataset Split")

    test_split = st.slider(
        "Test Set Ratio", 0.10, 0.50, 0.20, 0.05
    )

    seed = st.number_input(
        "Random Seed", value=42, step=1
    )

    st.markdown("### Tree Parameters")

    max_depth = st.slider(
        "Maximum Tree Depth", 2, 15, 5
    )

    min_split = st.slider(
        "Minimum Samples to Split", 2, 20, 2
    )

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if df_raw.empty or len(df_raw.columns) < 2:
            st.error(
                "Upload a non-empty CSV with at least "
                "one feature and one target column."
            )
            st.stop()

    except Exception as error:
        st.error(f"Unable to read CSV: {error}")
        st.stop()

    t1, t2, t3, t4 = st.tabs([
        "📁 Dataset Profile",
        "🤖 Model Studio",
        "📊 Diagnostics",
        "🧪 Prediction Sandbox"
    ])

    with t1:
        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_raw.shape[0])
        c2.metric("Columns", df_raw.shape[1])
        c3.metric(
            "Missing Values",
            int(df_raw.isnull().sum().sum())
        )

        st.dataframe(
            df_raw.head(),
            use_container_width=True
        )

        st.write("Column Data Types")
        st.write(df_raw.dtypes.astype(str))

    with t2:
        st.subheader("Model Training Workspace")

        target_col = st.selectbox(
            "Select Target Column",
            list(df_raw.columns),
            index=len(df_raw.columns) - 1
        )

        if st.button("🚀 Train Decision Tree Models"):
            try:
                prep = DataPreprocessor()

                df_imp = prep.impute_missing(
                    df_raw, target_col,
                    impute_num, impute_cat
                )

                df_enc, feature_cols = prep.encode_features(
                    df_imp, target_col, encoding_method
                )

                X_tr, X_te, y_tr, y_te = prep.split_data(
                    df_enc, target_col, feature_cols,
                    test_split, seed
                )

                models = {
                    "ID3": DecisionTreeStudioEngine(
                        "ID3", max_depth, min_split,
                        feature_cols
                    ).fit(X_tr, y_tr),

                    "C4.5": DecisionTreeStudioEngine(
                        "C4.5", max_depth, min_split,
                        feature_cols
                    ).fit(X_tr, y_tr)
                }

                st.session_state["data"] = {
                    "prep": prep,
                    "features": feature_cols,
                    "X_tr": X_tr,
                    "X_te": X_te,
                    "y_tr": y_tr,
                    "y_te": y_te
                }

                st.session_state["models"] = models

                st.success("Models trained successfully!")

            except Exception as error:
                st.error(f"Training failed: {error}")

        if st.session_state["models"]:
            selected_tree = st.radio(
                "Choose Tree",
                ["C4.5 Tree", "ID3 Tree"],
                horizontal=True
            )

            model_name = (
                "C4.5" if "C4.5" in selected_tree else "ID3"
            )

            st.graphviz_chart(
                st.session_state["models"][model_name].to_dot()
            )

    with t3:
        if (
            st.session_state["models"]
            and st.session_state["data"] is not None
        ):
            dp = st.session_state["data"]
            metrics = []

            for name, model in st.session_state["models"].items():
                predictions = model.predict(dp["X_te"])

                accuracy = accuracy_score(
                    dp["y_te"], predictions
                )

                precision, recall, f1, _ = (
                    precision_recall_fscore_support(
                        dp["y_te"],
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                metrics.append({
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1
                })

            metrics_df = pd.DataFrame(metrics).set_index("Model")

            st.subheader("Model Performance Comparison")
            st.dataframe(
                metrics_df.style.format("{:.4f}"),
                use_container_width=True
            )

            st.bar_chart(metrics_df)

            labels = np.arange(
                len(dp["prep"].target_encoder.classes_)
            )

            class_names = [
                str(x)
                for x in dp["prep"].target_encoder.classes_
            ]

            c1, c2 = st.columns(2)

            for col, name, cmap in [
                (c1, "ID3", "Blues"),
                (c2, "C4.5", "Oranges")
            ]:
                with col:
                    st.subheader(f"{name} Confusion Matrix")

                    predictions = st.session_state[
                        "models"
                    ][name].predict(dp["X_te"])

                    fig, ax = plt.subplots(figsize=(5, 4))

                    sns.heatmap(
                        confusion_matrix(
                            dp["y_te"],
                            predictions,
                            labels=labels
                        ),
                        annot=True,
                        fmt="d",
                        cmap=cmap,
                        cbar=False,
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Class")
                    ax.set_ylabel("Actual Class")

                    st.pyplot(fig)
                    plt.close(fig)

        else:
            st.info("Train the models to view diagnostics.")

    with t4:
        if (
            st.session_state["models"]
            and st.session_state["data"] is not None
        ):
            dp = st.session_state["data"]
            feature_cols = dp["features"]
            sandbox_input = {}

            st.subheader("Interactive Prediction Sandbox")

            for i, col in enumerate(feature_cols):
                values = dp["X_tr"][:, i]

                min_val = float(np.min(values))
                max_val = float(np.max(values))
                mean_val = float(np.mean(values))

                if min_val < max_val:
                    sandbox_input[col] = st.slider(
                        f"Adjust {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(
                            np.clip(mean_val, min_val, max_val)
                        ),
                        key=f"feature_{i}"
                    )
                else:
                    st.number_input(
                        f"Feature: {col}",
                        value=min_val,
                        disabled=True,
                        key=f"feature_{i}"
                    )
                    sandbox_input[col] = min_val

            if st.button("🔮 Predict Class"):
                row = np.array([[
                    sandbox_input[col]
                    for col in feature_cols
                ]])

                c1, c2 = st.columns(2)

                for col, name in [(c1, "ID3"), (c2, "C4.5")]:
                    prediction = int(
                        st.session_state["models"][name]
                        .predict(row)[0]
                    )

                    label = (
                        dp["prep"].target_encoder
                        .inverse_transform([prediction])[0]
                    )

                    col.metric(
                        f"{name} Prediction",
                        str(label)
                    )

        else:
            st.info("Train the models to enable predictions.")

else:
    st.info(
        "👋 Upload a classification CSV using the sidebar "
        "to get started."
    )
