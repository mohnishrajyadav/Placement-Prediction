"""
app.py
------
Flask entry point for the Placement Prediction dashboard.

Routes
------
/                 dashboard home
/data-loading     load & inspect the raw dataset
/eda              summary stats + charts (stub page for now)
"""
from ml.feature_engg import normalize, outlier
from flask import Flask, render_template, request
from ml.predict import predict_student, get_feature_info
from flask import Flask, render_template, jsonify

from ml.data_loader import load_data, clean_data, get_data_summary
from ml.eda import get_summary_stats, generate_charts

app = Flask(__name__)


@app.route("/")
def index():
    """Dashboard home page."""
    return render_template("index.html")


@app.route("/data-loading")
def data_loading():
    """
    Loads the raw dataset, cleans it, and shows before/after summary
    info (shape, dtypes, missing values, duplicates).
    """
    try:
        raw_df = load_data()
        raw_summary = get_data_summary(raw_df)

        cleaned_df = clean_data(raw_df)
        cleaned_summary = get_data_summary(cleaned_df)

        preview_rows = cleaned_df.head(10).to_dict(orient="records")
        preview_cols = list(cleaned_df.columns)

        return render_template(
            "data_loading.html",
            raw_summary=raw_summary,
            cleaned_summary=cleaned_summary,
            preview_rows=preview_rows,
            preview_cols=preview_cols,
            error=None,
        )
    except FileNotFoundError as e:
        return render_template(
            "data_loading.html",
            raw_summary=None,
            cleaned_summary=None,
            preview_rows=[],
            preview_cols=[],
            error=str(e),
        )


@app.route("/eda")
def eda():
    """
    Stub EDA page: runs summary stats + generates chart images into
    static/charts/, then renders them. Extend this as the project grows.
    """
    try:
        raw_df = load_data()
        cleaned_df = clean_data(raw_df)

        stats = get_summary_stats(cleaned_df)
        chart_files = generate_charts(cleaned_df)

        return render_template(
            "eda.html",
            stats=stats,
            chart_files=chart_files,
            error=None,
        )
    except FileNotFoundError as e:
        return render_template("eda.html", stats=None, chart_files=[], error=str(e))

@app.route("/feature-engg")
def feature_engg():
    """Feature engineering page."""

    try:
        # Load and clean the dataset
        df = load_data()
        cleaned_df = clean_data(df)

        # Apply feature engineering
        transformed_df = normalize(cleaned_df)

        # Detect outliers
        outlier_results = outlier(transformed_df)

        # Prepare preview
        preview_rows = transformed_df.head(10).to_dict(
            orient="records"
        )
        preview_cols = list(transformed_df.columns)

        return render_template(
            "feature_engg.html",
            preview_rows=preview_rows,
            preview_cols=preview_cols,
            outlier_results=outlier_results,
            error=None
        )

    except FileNotFoundError as e:
        return render_template(
            "feature_engg.html",
            preview_rows=[],
            preview_cols=[],
            outlier_results=None,
            error=str(e)
        )

@app.route("/api/data-summary")
def api_data_summary():
    """JSON endpoint for the raw data summary (handy for debugging/tests)."""
    raw_df = load_data()
    return jsonify(get_data_summary(raw_df))

@app.route("/predict", methods=["GET", "POST"])
def predict():

    error = None
    result = None

    try:
        feature_info = get_feature_info()

        numeric_cols = feature_info["numeric_cols"]
        categorical_cols = feature_info["categorical_cols"]

    except Exception as exc:
        return render_template(
            "predict.html",
            numeric_cols=[],
            categorical_cols=[],
            result=None,
            error=str(exc)
        )

    if request.method == "POST":

        try:
            student_data = {}

            # Read numerical input values
            for col in numeric_cols:
                value = request.form.get(col, "").strip()

                if not value:
                    raise ValueError(
                        f"Please provide a value for {col}."
                    )

                student_data[col] = float(value)

            # Read categorical input values
            for col in categorical_cols:
                value = request.form.get(col, "").strip()

                if not value:
                    raise ValueError(
                        f"Please provide a value for {col}."
                    )

                student_data[col] = value

            result = predict_student(student_data)

        except (ValueError, TypeError) as exc:
            error = str(exc)

        except Exception:
            error = (
                "Unable to generate the prediction. "
                "Check the model and submitted values."
            )

    return render_template(
        "predict.html",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        result=result,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
