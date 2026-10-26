import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        
    def impute_missing(self, df: pd.DataFrame, target_col: str, strategy_num: str = 'median', strategy_cat: str = 'mode') -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col == target_col:
                df = df.dropna(subset=[target_col])
                continue
                
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median() if strategy_num == 'median' else df[col].mean()
                    df[col] = df[col].fillna(val)
                else:
                    val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(val)
        return df

    def encode_features(self, df: pd.DataFrame, target_col: str, encoding_type: str = 'Label Encoding') -> tuple[pd.DataFrame, list]:
        df = df.copy()
        feature_cols = [c for c in df.columns if c != target_col]
        
        self.target_encoder = LabelEncoder()
        df[target_col] = self.target_encoder.fit_transform(df[target_col].astype(str))
        
        categorical_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        
        if encoding_type == 'One-Hot Encoding' and len(categorical_features) > 0:
            df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
            feature_cols = [c for c in df.columns if c != target_col]
        else:
            for col in categorical_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        return df, feature_cols

    def split_data(self, df: pd.DataFrame, target_col: str, feature_cols: list, test_size: float, random_state: int):
        X = df[feature_cols]
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
