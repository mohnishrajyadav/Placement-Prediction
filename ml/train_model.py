from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "placement_model.joblib"

TARGET = "PlacementStatus"


def train_model():

    # Find the dataset
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIR}"
        )

    data_path = next(
        (
            path for path in csv_files
            if "placement_predict_50k" in path.name.lower()
        ),
        csv_files[0]
    )

    df = pd.read_csv(data_path)

    print("Dataset loaded:", data_path.name)
    print("Dataset shape:", df.shape)

    # Validate target column
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove rows without a target
    df = df.dropna(subset=[TARGET]).copy()

    # Separate input features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Remove other target or outcome columns to prevent leakage
    other_targets = [
        "IsAnomaly",
        "Salary Package"
    ]

    X = X.drop(
        columns=[
            col for col in other_targets
            if col in X.columns
        ],
        errors="ignore"
    )

    # Remove identifiers that should not be used as predictors
    identifier_columns = [
        col for col in X.columns
        if col.lower() in {
            "studentid",
            "student_id",
            "id"
        }
    ]

    X = X.drop(columns=identifier_columns)

    # Require a binary classification target
    if y.nunique() != 2:
        raise ValueError(
            f"Expected a binary '{TARGET}' target, "
            f"but found {y.nunique()} classes: "
            f"{y.unique().tolist()}"
        )

    # Identify numeric and categorical features
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable input features found.")

    # Preprocess numerical features
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocess categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Combine preprocessing steps
    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers
    )

    # Build the machine learning pipeline
    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(max_iter=2000)
        )
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save the complete pipeline
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_cols": X.columns.tolist(),
            "target": TARGET
        },
        MODEL_PATH
    )

    print(f"\nModel saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()