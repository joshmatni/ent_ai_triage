from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["urgency"])
    y = df["urgency"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor(
    include_text: bool = True
) -> ColumnTransformer:

    # Define column groupings
    categorical_cols = ["nasal_discharge", "language"]
    boolean_cols = ["worsening", "fever", "dizziness", "hearing_change", "immunocompromised"]
    numeric_cols = ["duration_days", "pain_severity", "age"]
    text_cols = ["symptom_keywords"] if include_text else []

    transformers = []
    # One‑hot encode categorical variables
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    # Pass boolean variables through (they will be treated as 0/1)
    transformers.append(("bool", "passthrough", boolean_cols))
    # Standardize numerical variables
    transformers.append(("num", StandardScaler(), numeric_cols))

    if include_text:
        # Define a simple token pattern that splits on semicolons and whitespace
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(";"), preprocessor=None)
        transformers.append(("text", tfidf, text_cols))

    preprocessor: ColumnTransformer = ColumnTransformer(transformers, remainder="drop")
    return preprocessor


def preprocess_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple:
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_test_transformed


def encode_labels(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:

    unique_labels = sorted(y.unique())
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = y.map(mapping)
    return y_encoded, mapping