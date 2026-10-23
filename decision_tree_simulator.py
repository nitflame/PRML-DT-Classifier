# ================================================================
# COMPARATIVE DECISION TREE CLASSIFICATION SYSTEM
# Algorithms:
#   1. ID3   - Information Gain / Entropy
#   2. C4.5  - Gain Ratio
#   3. CART  - Gini Impurity
#   4. CHAID - Chi-Square Test
#
# Runs in VS Code or Google Colab.
# Input: CSV / Excel dataset
# Output:
#   - Validation messages
#   - Decision tree
#   - Confusion matrix
#   - Accuracy, Precision, Recall, F1
#   - Algorithm comparison
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2

# Numerical columns with more unique values than this are
# treated as continuous.
NUMERIC_UNIQUE_THRESHOLD = 10


# ================================================================
# GENERAL UTILITIES
# ================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_error(message):
    print(f"\n❌ ERROR: {message}")


def print_warning(message):
    print(f"\n⚠ WARNING: {message}")


def print_success(message):
    print(f"\n✓ {message}")


def entropy(y):
    """
    Entropy:
        H(S) = -Σ p_i log2(p_i)
    """
    if len(y) == 0:
        return 0.0

    counts = Counter(y)
    probabilities = np.array(list(counts.values())) / len(y)

    return -np.sum(
        probabilities * np.log2(probabilities + 1e-12)
    )


def gini_impurity(y):
    """
    Gini:
        Gini = 1 - Σ p_i²
    """
    if len(y) == 0:
        return 0.0

    counts = Counter(y)
    probabilities = np.array(list(counts.values())) / len(y)

    return 1.0 - np.sum(probabilities ** 2)


def majority_class(y):
    if len(y) == 0:
        return None

    return Counter(y).most_common(1)[0][0]


def weighted_impurity(groups, impurity_function):
    total = sum(len(group) for group in groups)

    if total == 0:
        return float("inf")

    value = 0.0

    for group in groups:
        if len(group) > 0:
            value += (len(group) / total) * impurity_function(group)

    return value


# ================================================================
# DATA LOADING
# ================================================================

def load_dataset():
    print_header("DATASET INPUT")

    path = input(
        "Enter dataset path (.csv / .xlsx / .xls): "
    ).strip()

    if not path:
        print_error("No file path was provided.")
        return None

    if not os.path.exists(path):
        print_error(f"File not found: {path}")
        return None

    try:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".csv":
            df = pd.read_csv(path)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        else:
            print_error(
                "Unsupported file format. Use CSV or Excel."
            )
            return None

    except Exception as e:
        print_error(f"Could not read dataset: {e}")
        return None

    return df


# ================================================================
# DATA VALIDATION
# ================================================================

def validate_dataset(df, target):
    """
    Validates the dataset before training.
    """

    print_header("DATASET VALIDATION")

    errors = []
    warnings_list = []

    # Empty dataset
    if df is None or df.empty:
        errors.append("The dataset is empty.")
        return False, errors, warnings_list

    # Number of rows
    if len(df) < 10:
        errors.append(
            f"Dataset contains only {len(df)} rows. "
            "At least 10 rows are recommended."
        )

    # Number of columns
    if df.shape[1] < 2:
        errors.append(
            "Dataset must contain at least one feature "
            "column and one target column."
        )

    # Target existence
    if target not in df.columns:
        errors.append(
            f"Target column '{target}' does not exist."
        )
        return False, errors, warnings_list

    # Duplicate columns
    if df.columns.duplicated().any():
        errors.append("Dataset contains duplicate column names.")

    # Completely empty columns
    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    if empty_columns:
        errors.append(
            "Completely empty columns found: "
            + ", ".join(empty_columns)
        )

    # Target missing values
    target_missing = df[target].isna().sum()

    if target_missing > 0:
        warnings_list.append(
            f"Target column contains {target_missing} missing values. "
            "Rows with missing target values will be removed."
        )

    # Target classes
    target_clean = df[target].dropna()

    if target_clean.nunique() < 2:
        errors.append(
            "Target column contains fewer than two classes. "
            "A classification tree requires at least two classes."
        )

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        warnings_list.append(
            "Missing values detected. Numeric values will use "
            "median imputation and categorical values will use mode."
        )

    # Constant columns
    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1
        and col != target
    ]

    if constant_columns:
        warnings_list.append(
            "Constant feature columns will be removed: "
            + ", ".join(constant_columns)
        )

    # Infinite numerical values
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    infinite_columns = []

    for col in numeric_cols:
        if np.isinf(df[col]).any():
            infinite_columns.append(col)

    if infinite_columns:
        errors.append(
            "Infinite numerical values found in: "
            + ", ".join(infinite_columns)
        )

    # Print results
    if errors:
        for error in errors:
            print_error(error)

        return False, errors, warnings_list

    print_success("Dataset structure is valid.")

    for warning in warnings_list:
        print_warning(warning)

    print_success(
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    )

    print_success(
        f"Target classes: {df[target].nunique()}"
    )

    return True, errors, warnings_list


# ================================================================
# PREPROCESSING
# ================================================================

def preprocess_data(df, target):
    """
    Converts categorical values to integer codes.

    Returns:
        X
        y
        feature_names
        encoders
    """

    print_header("DATA PREPROCESSING")

    data = df.copy()

    # Remove rows with missing target
    data = data.dropna(subset=[target])

    # Remove completely empty columns
    data = data.dropna(axis=1, how="all")

    # Remove constant columns except target
    for col in list(data.columns):
        if col != target and data[col].nunique() <= 1:
            data.drop(columns=[col], inplace=True)

    feature_columns = [
        col for col in data.columns
        if col != target
    ]

    X = data[feature_columns].copy()
    y = data[target].copy()

    encoders = {}

    # Feature preprocessing
    for col in feature_columns:

        if pd.api.types.is_numeric_dtype(X[col]):

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            median_value = X[col].median()

            if pd.isna(median_value):
                median_value = 0

            X[col] = X[col].fillna(median_value)

        else:

            X[col] = X[col].astype(str)

            mode = X[col].mode()

            fill_value = mode.iloc[0] if len(mode) else "Unknown"

            X[col] = X[col].replace(
                ["nan", "None", ""],
                np.nan
            )

            X[col] = X[col].fillna(fill_value)

            encoder = LabelEncoder()

            X[col] = encoder.fit_transform(
                X[col].astype(str)
            )

            encoders[col] = encoder

    # Encode target
    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        y.astype(str)
    )

    encoders["__target__"] = target_encoder

    print_success(
        f"Features after preprocessing: {len(feature_columns)}"
    )

    print_success(
        f"Classes: {list(target_encoder.classes_)}"
    )

    return (
        X.reset_index(drop=True),
        pd.Series(y),
        feature_columns,
        encoders
    )


# ================================================================
# TREE NODE
# ================================================================

class TreeNode:

    def __init__(
        self,
        prediction=None,
        feature=None,
        threshold=None,
        branches=None,
        is_leaf=False,
        class_counts=None,
        depth=0,
        rule=None
    ):

        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.branches = branches or {}
        self.is_leaf = is_leaf
        self.class_counts = class_counts
        self.depth = depth
        self.rule = rule


# ================================================================
# BASE DECISION TREE
# ================================================================

class DecisionTreeBase:

    def __init__(
        self,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def stopping_condition(self, y, depth):

        if len(y) == 0:
            return True

        if len(set(y)) == 1:
            return True

        if depth >= self.max_depth:
            return True

        if len(y) < self.min_samples_split:
            return True

        return False

    def create_leaf(self, y, depth):

        return TreeNode(
            prediction=majority_class(y),
            is_leaf=True,
            class_counts=Counter(y),
            depth=depth
        )

    def predict_one(self, row, node):

        if node.is_leaf:
            return node.prediction

        value = row[node.feature]

        # Binary split
        if node.threshold is not None:

            if value <= node.threshold:
                branch = "left"
            else:
                branch = "right"

            child = node.branches.get(branch)

            if child is None:
                return node.prediction

            return self.predict_one(row, child)

        # Multiway split
        else:

            key = value

            child = node.branches.get(key)

            if child is None:
                return node.prediction

            return self.predict_one(row, child)

    def predict(self, X):

        return np.array([
            self.predict_one(row, self.root)
            for _, row in X.iterrows()
        ])

    def get_class_counts(self, y):
        return Counter(y)


# ================================================================
# SPLIT UTILITIES
# ================================================================

def numerical_thresholds(values):

    unique_values = sorted(
        np.unique(values)
    )

    if len(unique_values) <= 1:
        return []

    thresholds = []

    for i in range(len(unique_values) - 1):

        threshold = (
            unique_values[i] +
            unique_values[i + 1]
        ) / 2

        thresholds.append(threshold)

    return thresholds


def split_binary(X, y, feature, threshold):

    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold

    left_y = y[left_mask]
    right_y = y[right_mask]

    return left_mask, right_mask, left_y, right_y


# ================================================================
# ID3
# ================================================================

class ID3Tree(DecisionTreeBase):

    def information_gain(self, y, groups):

        parent_entropy = entropy(y)

        weighted_entropy = 0.0

        for group in groups:

            if len(group) == 0:
                continue

            weighted_entropy += (
                len(group) / len(y)
            ) * entropy(group)

        return parent_entropy - weighted_entropy

    def best_split(self, X, y, features):

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in features:

            values = X[feature]

            if pd.api.types.is_numeric_dtype(values):

                thresholds = numerical_thresholds(values)

                for threshold in thresholds:

                    left_mask, right_mask, left_y, right_y = (
                        split_binary(
                            X,
                            y,
                            feature,
                            threshold
                        )
                    )

                    if (
                        len(left_y) < self.min_samples_leaf
                        or
                        len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    gain = self.information_gain(
                        y,
                        [left_y, right_y]
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

            else:

                groups = []

                for value in values.unique():

                    group_y = y[values == value]

                    if len(group_y) >= self.min_samples_leaf:
                        groups.append(group_y)

                if len(groups) < 2:
                    continue

                gain = self.information_gain(
                    y,
                    groups
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = None

        return best_feature, best_threshold, best_gain

    def build(self, X, y, features, depth=0):

        if self.stopping_condition(y, depth):

            return self.create_leaf(y, depth)

        feature, threshold, gain = self.best_split(
            X,
            y,
            features
        )

        if feature is None or gain <= 1e-10:

            return self.create_leaf(y, depth)

        node = TreeNode(
            prediction=majority_class(y),
            feature=feature,
            threshold=threshold,
            class_counts=Counter(y),
            depth=depth
        )

        if threshold is not None:

            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold

            node.rule = (
                f"{feature} <= {threshold:.3f}"
            )

            node.branches["left"] = self.build(
                X[left_mask],
                y[left_mask],
                features,
                depth + 1
            )

            node.branches["right"] = self.build(
                X[right_mask],
                y[right_mask],
                features,
                depth + 1
            )

        else:

            node.rule = f"{feature}"

            for value in X[feature].unique():

                mask = X[feature] == value

                node.branches[value] = self.build(
                    X[mask],
                    y[mask],
                    [
                        f for f in features
                        if f != feature
                    ],
                    depth + 1
                )

        return node

    def fit(self, X, y):

        self.root = self.build(
            X,
            y,
            list(X.columns)
        )

        return self


# ================================================================
# C4.5
# ================================================================

class C45Tree(ID3Tree):

    def gain_ratio(
        self,
        y,
        groups
    ):

        gain = self.information_gain(
            y,
            groups
        )

        split_info = 0.0

        for group in groups:

            if len(group) == 0:
                continue

            probability = len(group) / len(y)

            split_info -= (
                probability *
                np.log2(probability + 1e-12)
            )

        if split_info == 0:
            return 0.0

        return gain / split_info

    def best_split(self, X, y, features):

        best_ratio = -1
        best_feature = None
        best_threshold = None

        for feature in features:

            values = X[feature]

            if pd.api.types.is_numeric_dtype(values):

                thresholds = numerical_thresholds(values)

                for threshold in thresholds:

                    left_mask, right_mask, left_y, right_y = (
                        split_binary(
                            X,
                            y,
                            feature,
                            threshold
                        )
                    )

                    if (
                        len(left_y) < self.min_samples_leaf
                        or
                        len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    groups = [
                        left_y,
                        right_y
                    ]

                    ratio = self.gain_ratio(
                        y,
                        groups
                    )

                    if ratio > best_ratio:

                        best_ratio = ratio
                        best_feature = feature
                        best_threshold = threshold

            else:

                groups = []

                for value in values.unique():

                    group_y = y[values == value]

                    if len(group_y) >= self.min_samples_leaf:
                        groups.append(group_y)

                if len(groups) < 2:
                    continue

                ratio = self.gain_ratio(
                    y,
                    groups
                )

                if ratio > best_ratio:

                    best_ratio = ratio
                    best_feature = feature
                    best_threshold = None

        return (
            best_feature,
            best_threshold,
            best_ratio
        )

    def fit(self, X, y):

        self.root = self.build(
            X,
            y,
            list(X.columns)
        )

        return self


# ================================================================
# CART
# ================================================================

class CARTTree(DecisionTreeBase):

    def best_split(self, X, y, features):

        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in features:

            values = X[feature]

            # CART creates binary splits
            if pd.api.types.is_numeric_dtype(values):

                thresholds = numerical_thresholds(values)

                for threshold in thresholds:

                    left_mask, right_mask, left_y, right_y = (
                        split_binary(
                            X,
                            y,
                            feature,
                            threshold
                        )
                    )

                    if (
                        len(left_y) < self.min_samples_leaf
                        or
                        len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    gini = weighted_impurity(
                        [left_y, right_y],
                        gini_impurity
                    )

                    if gini < best_gini:

                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold

            else:

                # One-vs-rest binary splits for categorical data
                unique_values = values.unique()

                for value in unique_values:

                    left_mask = values == value
                    right_mask = values != value

                    left_y = y[left_mask]
                    right_y = y[right_mask]

                    if (
                        len(left_y) < self.min_samples_leaf
                        or
                        len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    gini = weighted_impurity(
                        [left_y, right_y],
                        gini_impurity
                    )

                    if gini < best_gini:

                        best_gini = gini
                        best_feature = feature
                        best_threshold = value

        return (
            best_feature,
            best_threshold,
            best_gini
        )

    def build(self, X, y, depth=0):

        if self.stopping_condition(y, depth):

            return self.create_leaf(y, depth)

        features = list(X.columns)

        feature, threshold, score = self.best_split(
            X,
            y,
            features
        )

        if feature is None:

            return self.create_leaf(y, depth)

        values = X[feature]

        if pd.api.types.is_numeric_dtype(values):

            left_mask = values <= threshold
            right_mask = values > threshold

            rule = (
                f"{feature} <= {threshold:.3f}"
            )

        else:

            left_mask = values == threshold
            right_mask = values != threshold

            rule = (
                f"{feature} == {threshold}"
            )

        if (
            left_mask.sum() < self.min_samples_leaf
            or
            right_mask.sum() < self.min_samples_leaf
        ):

            return self.create_leaf(y, depth)

        node = TreeNode(
            prediction=majority_class(y),
            feature=feature,
            threshold=threshold,
            class_counts=Counter(y),
            depth=depth,
            rule=rule
        )

        node.branches["left"] = self.build(
            X[left_mask],
            y[left_mask],
            depth + 1
        )

        node.branches["right"] = self.build(
            X[right_mask],
            y[right_mask],
            depth + 1
        )

        return node

    def fit(self, X, y):

        self.root = self.build(X, y)

        return self


# ================================================================
# CHAID
# ================================================================

class CHAIDTree(DecisionTreeBase):

    def chi_square_score(
        self,
        feature_values,
        y
    ):

        table = pd.crosstab(
            feature_values,
            y
        )

        if table.shape[0] < 2 or table.shape[1] < 2:
            return 1.0

        try:

            chi2, p_value, _, _ = chi2_contingency(
                table
            )

            return p_value

        except Exception:
            return 1.0

    def merge_categories(
        self,
        feature_values,
        y
    ):
        """
        Basic CHAID-style category merging.

        Categories that are statistically similar are grouped.
        """

        categories = list(
            feature_values.unique()
        )

        if len(categories) <= 2:
            return categories

        changed = True

        while changed and len(categories) > 2:

            changed = False

            best_pair = None
            highest_p = -1

            for i in range(len(categories)):

                for j in range(
                    i + 1,
                    len(categories)
                ):

                    a = categories[i]
                    b = categories[j]

                    mask = feature_values.isin(
                        [a, b]
                    )

                    values = feature_values[mask]
                    labels = y[mask]

                    table = pd.crosstab(
                        values,
                        labels
                    )

                    if table.shape[0] < 2:
                        continue

                    try:

                        _, p, _, _ = chi2_contingency(
                            table
                        )

                        if p > highest_p:

                            highest_p = p
                            best_pair = (a, b)

                    except Exception:
                        pass

            # Merge categories if they are similar
            if best_pair is not None and highest_p > 0.05:

                a, b = best_pair

                categories.remove(a)
                categories.remove(b)

                categories.append(
                    (a, b)
                )

                changed = True

            else:
                break

        return categories

    def best_split(
        self,
        X,
        y
    ):

        best_feature = None
        best_p = 1.0

        for feature in X.columns:

            values = X[feature]

            # For numerical variables, create bins.
            if pd.api.types.is_numeric_dtype(values):

                try:

                    binned = pd.qcut(
                        values,
                        q=min(4, values.nunique()),
                        duplicates="drop"
                    )

                    p = self.chi_square_score(
                        binned,
                        y
                    )

                except Exception:
                    continue

            else:

                p = self.chi_square_score(
                    values,
                    y
                )

            # Lower p-value = stronger association
            if p < best_p:

                best_p = p
                best_feature = feature

        return best_feature, best_p

    def build(self, X, y, depth=0):

        if self.stopping_condition(y, depth):

            return self.create_leaf(y, depth)

        feature, p_value = self.best_split(
            X,
            y
        )

        # CHAID significance threshold
        if feature is None or p_value > 0.05:

            return self.create_leaf(y, depth)

        node = TreeNode(
            prediction=majority_class(y),
            feature=feature,
            class_counts=Counter(y),
            depth=depth,
            rule=f"{feature} (Chi-square p={p_value:.4f})"
        )

        values = X[feature]

        # Numerical values are grouped into quantile bins
        if pd.api.types.is_numeric_dtype(values):

            try:

                categories = pd.qcut(
                    values,
                    q=min(
                        4,
                        values.nunique()
                    ),
                    duplicates="drop"
                )

                for category in categories.unique():

                    mask = categories == category

                    if mask.sum() < self.min_samples_leaf:
                        continue

                    node.branches[str(category)] = self.build(
                        X[mask].drop(columns=[feature]),
                        y[mask],
                        depth + 1
                    )

            except Exception:

                return self.create_leaf(y, depth)

        else:

            for value in values.unique():

                mask = values == value

                if mask.sum() < self.min_samples_leaf:
                    continue

                node.branches[value] = self.build(
                    X[mask].drop(columns=[feature]),
                    y[mask],
                    depth + 1
                )

        if len(node.branches) < 2:

            return self.create_leaf(y, depth)

        return node

    def predict_one(self, row, node):

        if node.is_leaf:
            return node.prediction

        feature = node.feature

        if feature not in row.index:
            return node.prediction

        value = row[feature]

        # Numerical CHAID
        if node.branches:

            # Exact category
            if value in node.branches:

                return self.predict_one(
                    row,
                    node.branches[value]
                )

            # For numerical bins, determine closest branch
            if pd.api.types.is_numeric_dtype(
                type(value)
            ):
                pass

        return node.prediction

    def predict(self, X):

        predictions = []

        for _, row in X.iterrows():

            try:
                prediction = self.predict_one(
                    row,
                    self.root
                )
            except Exception:
                prediction = self.root.prediction

            predictions.append(prediction)

        return np.array(predictions)

    def fit(self, X, y):

        self.root = self.build(
            X,
            y
        )

        return self


# ================================================================
# TREE DISPLAY
# ================================================================

def print_tree(node, feature_names, indent="", branch_label="ROOT"):

    if node is None:
        return

    if node.is_leaf:

        print(
            f"{indent}{branch_label} → "
            f"CLASS {node.prediction} "
            f"{dict(node.class_counts)}"
        )

        return

    print(
        f"{indent}{branch_label} → "
        f"[{node.rule}]"
    )

    items = list(node.branches.items())

    for i, (branch, child) in enumerate(items):

        is_last = i == len(items) - 1

        connector = "└── " if is_last else "├── "

        print(
            f"{indent}{connector}{branch}"
        )

        child_indent = (
            indent + "    "
            if is_last
            else indent + "│   "
        )

        print_tree(
            child,
            feature_names,
            child_indent,
            "→"
        )


# ================================================================
# TREE VISUALIZATION
# ================================================================

def collect_tree_lines(
    node,
    prefix="",
    is_last=True,
    label="ROOT"
):

    lines = []

    if node.is_leaf:

        lines.append(
            (
                prefix,
                f"{label}: CLASS {node.prediction}"
            )
        )

        return lines

    lines.append(
        (
            prefix,
            f"{label}: {node.rule}"
        )
    )

    children = list(node.branches.items())

    for i, (branch, child) in enumerate(children):

        last = i == len(children) - 1

        connector = "└── " if last else "├── "

        child_prefix = (
            prefix +
            ("    " if last else "│   ")
        )

        lines.append(
            (
                prefix + connector,
                f"{branch}"
            )
        )

        lines.extend(
            collect_tree_lines(
                child,
                child_prefix,
                last,
                "→"
            )
        )

    return lines


def visualize_tree(tree, algorithm):

    print_header(
        f"{algorithm} DECISION TREE"
    )

    print_tree(
        tree.root,
        []
    )

    # Generate a simple visual diagram using matplotlib
    lines = collect_tree_lines(tree.root)

    height = max(
        4,
        len(lines) * 0.45
    )

    fig, ax = plt.subplots(
        figsize=(14, height)
    )

    ax.axis("off")

    y = 1.0

    step = 1.0 / max(
        len(lines),
        1
    )

    for prefix, text in lines:

        ax.text(
            0.02,
            y,
            prefix + text,
            fontsize=9,
            family="monospace",
            verticalalignment="top"
        )

        y -= step

    ax.set_title(
        f"{algorithm} Decision Tree",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# EVALUATION
# ================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    class_names,
    algorithm,
    show_matrix=True
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print_header(
        f"{algorithm} PERFORMANCE"
    )

    print(
        f"Accuracy   : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision  : {precision * 100:.2f}%"
    )

    print(
        f"Recall     : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score   : {f1 * 100:.2f}%"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    if show_matrix:

        try:

            display_labels = [
                str(x)
                for x in class_names
            ]

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels
            )

            disp.plot(
                cmap="Blues",
                values_format="d"
            )

            plt.title(
                f"{algorithm} - Confusion Matrix"
            )

            plt.tight_layout()
            plt.show()

        except Exception:
            pass

    return {
        "Algorithm": algorithm,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

def prepare_split(X, y):

    print_header("TRAIN / TEST SPLIT")

    class_counts = Counter(y)

    # Stratification is only possible when each class
    # has at least two samples.
    can_stratify = all(
        count >= 2
        for count in class_counts.values()
    )

    try:

        if can_stratify:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y
                )
            )

        else:

            print_warning(
                "Some classes contain fewer than 2 samples. "
                "Using non-stratified split."
            )

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE
                )
            )

    except Exception as e:

        print_error(
            f"Could not create train/test split: {e}"
        )

        return None

    print_success(
        f"Training samples: {len(X_train)}"
    )

    print_success(
        f"Testing samples: {len(X_test)}"
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )


# ================================================================
# MODEL CREATION
# ================================================================

def create_model(choice):

    if choice == "1":

        return ID3Tree(), "ID3"

    elif choice == "2":

        return C45Tree(), "C4.5"

    elif choice == "3":

        return CARTTree(), "CART"

    elif choice == "4":

        return CHAIDTree(), "CHAID"

    return None, None


# ================================================================
# ALGORITHM MENU
# ================================================================

def algorithm_menu():

    print_header(
        "DECISION TREE ALGORITHM SELECTION"
    )

    print("1. ID3   - Information Gain / Entropy")
    print("2. C4.5  - Gain Ratio")
    print("3. CART  - Gini Impurity")
    print("4. CHAID - Chi-Square")
    print("5. ALL   - Compare all algorithms")

    while True:

        choice = input(
            "\nSelect algorithm [1-5]: "
        ).strip()

        if choice in ["1", "2", "3", "4", "5"]:
            return choice

        print_error(
            "Invalid choice. Please enter 1, 2, 3, 4 or 5."
        )


# ================================================================
# RUN ONE ALGORITHM
# ================================================================

def run_algorithm(
    choice,
    X_train,
    X_test,
    y_train,
    y_test,
    class_names
):

    model, algorithm = create_model(
        choice
    )

    if model is None:
        return None

    print_header(
        f"TRAINING {algorithm}"
    )

    print(
        "Building decision tree..."
    )

    try:

        model.fit(
            X_train,
            y_train
        )

    except Exception as e:

        print_error(
            f"{algorithm} training failed: {e}"
        )

        return None

    print_success(
        f"{algorithm} tree created successfully."
    )

    visualize_tree(
        model,
        algorithm
    )

    result = evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        algorithm
    )

    return result


# ================================================================
# COMPARISON TABLE
# ================================================================

def show_comparison(results):

    if not results:
        return

    print_header(
        "ALGORITHM COMPARISON"
    )

    table = pd.DataFrame(
        results
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        table[column] = (
            table[column] * 100
        ).round(2)

    print(
        table.to_string(
            index=False
        )
    )

    best_index = table[
        "Accuracy"
    ].idxmax()

    best_algorithm = table.loc[
        best_index,
        "Algorithm"
    ]

    best_accuracy = table.loc[
        best_index,
        "Accuracy"
    ]

    print(
        f"\n🏆 Best Algorithm: "
        f"{best_algorithm}"
    )

    print(
        f"   Accuracy: {best_accuracy:.2f}%"
    )

    # Comparison graph
    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        table["Algorithm"],
        table["Accuracy"]
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.title(
        "Decision Tree Algorithm Comparison"
    )

    plt.ylim(
        0,
        100
    )

    for i, value in enumerate(
        table["Accuracy"]
    ):

        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.show()


# ================================================================
# DATASET INFORMATION
# ================================================================

def show_dataset_info(
    df,
    target
):

    print_header(
        "DATASET INFORMATION"
    )

    print(
        f"Rows          : {df.shape[0]}"
    )

    print(
        f"Columns       : {df.shape[1]}"
    )

    print(
        f"Target        : {target}"
    )

    print(
        f"Target classes: {df[target].nunique()}"
    )

    print("\nColumns:")

    for col in df.columns:

        dtype = str(
            df[col].dtype
        )

        unique = df[col].nunique(
            dropna=True
        )

        missing = df[col].isna().sum()

        print(
            f"  • {col:<25} "
            f"type={dtype:<12} "
            f"unique={unique:<5} "
            f"missing={missing}"
        )

    print("\nFirst 5 rows:")

    print(
        df.head().to_string(
            index=False
        )
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    print_header(
        "COMPARATIVE DECISION TREE CLASSIFICATION SYSTEM"
    )

    print(
        "Algorithms: ID3 | C4.5 | CART | CHAID"
    )

    print(
        "Dataset validation + preprocessing + "
        "tree generation + evaluation"
    )

    # ------------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------------

    df = load_dataset()

    if df is None:
        return

    # Remove whitespace from column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # ------------------------------------------------------------
    # TARGET SELECTION
    # ------------------------------------------------------------

    print_header(
        "TARGET COLUMN SELECTION"
    )

    print(
        "Available columns:"
    )

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i}. {column}"
        )

    target_input = input(
        "\nEnter target column name: "
    ).strip()

    # Allow column number too
    if target_input.isdigit():

        index = int(target_input) - 1

        if (
            0 <= index < len(df.columns)
        ):
            target = df.columns[index]

        else:
            print_error(
                "Invalid target column number."
            )
            return

    else:

        target = target_input

    # ------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------

    valid, errors, warnings_list = (
        validate_dataset(
            df,
            target
        )
    )

    if not valid:

        print_header(
            "PROCESS STOPPED"
        )

        print(
            "Please correct the dataset and "
            "run the program again."
        )

        return

    # ------------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------------

    show_dataset_info(
        df,
        target
    )

    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    X, y, feature_names, encoders = (
        preprocess_data(
            df,
            target
        )
    )

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    split = prepare_split(
        X,
        y
    )

    if split is None:
        return

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split

    # ------------------------------------------------------------
    # SELECT ALGORITHM
    # ------------------------------------------------------------

    choice = algorithm_menu()

    class_names = encoders[
        "__target__"
    ].classes_

    results = []

    # ------------------------------------------------------------
    # SINGLE ALGORITHM
    # ------------------------------------------------------------

    if choice in ["1", "2", "3", "4"]:

        result = run_algorithm(
            choice,
            X_train,
            X_test,
            y_train,
            y_test,
            class_names
        )

        if result:
            results.append(result)

    # ------------------------------------------------------------
    # ALL ALGORITHMS
    # ------------------------------------------------------------

    elif choice == "5":

        for algorithm_choice in [
            "1",
            "2",
            "3",
            "4"
        ]:

            result = run_algorithm(
                algorithm_choice,
                X_train,
                X_test,
                y_train,
                y_test,
                class_names
            )

            if result:
                results.append(result)

        show_comparison(
            results
        )

    # ------------------------------------------------------------
    # FINAL MESSAGE
    # ------------------------------------------------------------

    print_header(
        "SIMULATION COMPLETED"
    )

    print(
        "✓ Dataset validated"
    )

    print(
        "✓ Data preprocessing completed"
    )

    print(
        "✓ Decision tree generated"
    )

    print(
        "✓ Confusion matrix generated"
    )

    print(
        "✓ Performance metrics calculated"
    )

    if choice == "5":

        print(
            "✓ All four algorithms compared"
        )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()