from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"


def load_model():
    """Load the trained placement prediction pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run: python -m ml.train_model"
        )

    return joblib.load(MODEL_PATH)


def get_feature_info():
    """Return the input features required by the model."""

    bundle = load_model()

    return {
        "numeric_cols": bundle["numeric_cols"],
        "categorical_cols": bundle["categorical_cols"],
        "feature_cols": bundle["feature_cols"]
    }


def predict_student(student_data):
    """
    Predict placement for one student.

    student_data should be a dictionary containing
    the student's feature values.
    """

    bundle = load_model()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Arrange values in the original training column order
    student_df = pd.DataFrame(
        [student_data]
    )

    student_df = student_df.reindex(
        columns=feature_cols
    )

    # Predict placement class
    prediction = model.predict(student_df)[0]

    result = {
        "prediction": str(prediction),
        "confidence": None
    }

    # Get the probability of the predicted class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student_df)[0]

        result["confidence"] = round(
            float(max(probabilities)) * 100,
            2
        )

    return result