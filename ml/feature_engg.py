"""
ml/feature_engg.py
------------------
Feature Engineering Module

Performs preprocessing, creates additional features,
and provides a Random Forest classification experiment.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

from ml.data_loader import load_data


# --------------------------------------------------
# 1. FEATURE ENGINEERING
# --------------------------------------------------

def perform_feature_engineering(df: pd.DataFrame):
    """
    Create additional features and return:
    1. Processed DataFrame
    2. Summary of newly created features
    """

    engineered_df = df.copy()

    # Academic Score
    academic_cols = [
        col for col in [
            "CGPA",
            "SSC_Marks",
            "HSC_Marks"
        ]
        if col in engineered_df.columns
    ]

    if academic_cols:
        engineered_df["AcademicScore"] = (
            engineered_df[academic_cols].mean(axis=1)
        )

    # Experience Score
    experience_cols = [
        col for col in [
            "Internships",
            "Projects",
            "Workshops/Certifications"
        ]
        if col in engineered_df.columns
    ]

    if experience_cols:
        engineered_df["ExperienceScore"] = (
            engineered_df[experience_cols].sum(axis=1)
        )

    # Skill Score
    skill_cols = [
        col for col in [
            "AptitudeTestScore",
            "SoftSkillsRating"
        ]
        if col in engineered_df.columns
    ]

    if skill_cols:
        engineered_df["SkillScore"] = (
            engineered_df[skill_cols].mean(axis=1)
        )

    # Total Profile Score
    total_cols = [
        col for col in [
            "AcademicScore",
            "ExperienceScore",
            "SkillScore"
        ]
        if col in engineered_df.columns
    ]

    if total_cols:
        engineered_df["ProfileScore"] = (
            engineered_df[total_cols].mean(axis=1)
        )

    # Summarize newly created features
    new_features = [
        col for col in engineered_df.columns
        if col not in df.columns
    ]

    summary = {}

    for col in new_features:
        summary[col] = {
            "Mean": round(engineered_df[col].mean(), 2),
            "Min": round(engineered_df[col].min(), 2),
            "Max": round(engineered_df[col].max(), 2)
        }

    return engineered_df, summary


# --------------------------------------------------
# 2. NORMALIZATION
# --------------------------------------------------

def normalize(df, exclude_cols=None):
    """
    Normalize numerical input columns using min-max scaling.
    Excluded columns are left unchanged.
    """

    from sklearn.preprocessing import MinMaxScaler

    result = df.copy()
    exclude_cols = exclude_cols or []

    numeric_cols = result.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cols_to_scale = [
        col for col in numeric_cols
        if col not in exclude_cols
    ]

    if cols_to_scale:
        scaler = MinMaxScaler()

        result[cols_to_scale] = scaler.fit_transform(
            result[cols_to_scale]
        )

    return result


# --------------------------------------------------
# 3. OUTLIER DETECTION
# --------------------------------------------------

def outlier(df):
    """
    Detect outliers in numerical columns using the IQR method.

    Returns a dictionary containing outlier counts by column.
    """

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    results = {}

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        count = (
            (numeric_df[col] < lower_bound)
            | (numeric_df[col] > upper_bound)
        ).sum()

        results[col] = int(count)

    return results


# --------------------------------------------------
# 4. SIMPLE LINEAR REGRESSION
# --------------------------------------------------

def simple_linear_regression(df):
    """
    Predict SalaryPackage using CGPA, when both columns exist.

    Returns evaluation metrics instead of printing them.
    """

    required_cols = ["CGPA", "SalaryPackage"]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df[required_cols].dropna()

    if len(data) < 2:
        raise ValueError(
            "Not enough complete rows for regression."
        )

    X = data[["CGPA"]]
    y = data["SalaryPackage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "model": model,
        "mse": float(mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions))
    }


# --------------------------------------------------
# 5. PREPARE RANDOM FOREST FEATURES
# --------------------------------------------------

def prepare_data(df):
    """
    Prepare features and target for placement classification.
    """

    feature_cols = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops"
    ]

    target_col = "PlacementStatus"

    missing = [
        col for col in feature_cols + [target_col]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}"
        )

    data = df[feature_cols + [target_col]].dropna().copy()

    X = data[feature_cols]
    y = data[target_col]

    return X, y


# --------------------------------------------------
# 6. RANDOM FOREST CLASSIFIER
# --------------------------------------------------

def train_random_forest(df):
    """
    Train and evaluate a Random Forest classifier.

    Returns the trained model and evaluation results.
    """

    X, y = prepare_data(df)

    if y.nunique() < 2:
        raise ValueError(
            "Random Forest classification requires at least "
            "two target classes."
        )

    # Stratification requires at least two rows per class.
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each placement class needs at least two records "
            "for a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return {
        "model": model,
        "accuracy": float(accuracy),
        "classification_report": report
    }


# --------------------------------------------------
# 7. RUN EXPERIMENTS ONLY WHEN EXECUTED DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    engineered_df, summary = perform_feature_engineering(df)

    print("\nFeature Engineering Summary")
    print(summary)

    print("\nRandom Forest Classification")

    rf_results = train_random_forest(engineered_df)

    print("Accuracy:", rf_results["accuracy"])
    print(rf_results["classification_report"])

    # Run regression only if the required columns exist.
    if "SalaryPackage" in df.columns and "CGPA" in df.columns:
        print("\nLinear Regression")

        regression_results = simple_linear_regression(df)

        print("MSE:", regression_results["mse"])
        print("R²:", regression_results["r2"])