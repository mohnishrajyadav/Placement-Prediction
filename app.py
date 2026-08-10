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


@app.route("/api/data-summary")
def api_data_summary():
    """JSON endpoint for the raw data summary (handy for debugging/tests)."""
    raw_df = load_data()
    return jsonify(get_data_summary(raw_df))


if __name__ == "__main__":
    app.run(debug=True)
