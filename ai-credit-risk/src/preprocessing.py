"""
src/preprocessing.py

Builds a scikit-learn ColumnTransformer for numeric + categorical
features. The transformer is always FIT on training data only, then
reused (never refit) on test data, scenario inputs, and live predictions
- this is what prevents data leakage throughout the platform.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_feature_types(df, feature_columns):
    numeric_cols = [c for c in feature_columns if str(df[c].dtype) in
                    ("int64", "float64", "int32", "float32")]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(df, feature_columns):
    numeric_cols, categorical_cols = split_feature_types(df, feature_columns)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def get_output_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Human-readable feature names after the ColumnTransformer expands
    one-hot columns - used by explainability to label SHAP/importance
    output meaningfully."""
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols)
        names.extend(cat_names.tolist())
    return names
