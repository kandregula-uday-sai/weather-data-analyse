"""Simple Flask dashboard for weather data exploration and rain prediction."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_data_shuffled.csv"
MODEL_PATH = BASE_DIR / "models" / "rain_prediction_model.pkl"
RESULTS_PATH = BASE_DIR / "model_results.json"
CHART_DIR = BASE_DIR / "static" / "charts"
TARGET = "RainTomorrow"

app = Flask(__name__)
app.config["SECRET_KEY"] = "mca-weather-project-local-key"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Keep the CSV in the project folder.")
DATA = pd.read_csv(DATA_PATH)
if TARGET not in DATA.columns:
    raise ValueError(f"Dataset must contain the '{TARGET}' column.")
FEATURE_NAMES = [column for column in DATA.columns if column != TARGET]

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run: python train.py")
with MODEL_PATH.open("rb") as model_file:
    MODEL = pickle.load(model_file)

try:
    MODEL_RESULTS = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    MODEL_RESULTS = None

DIRECTION_FEATURES = {"WindGustDir", "WindDir9am", "WindDir3pm"}
BASIC_FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Sunshine", "Humidity3pm", "RainToday"]
FEATURE_INFO = {
    "MinTemp": ("Minimum temperature", "°C", "Lowest temperature recorded."),
    "MaxTemp": ("Maximum temperature", "°C", "Highest temperature recorded."),
    "Rainfall": ("Rainfall", "mm", "Rain measured for the day."),
    "Evaporation": ("Evaporation", "mm", "Water lost through evaporation."),
    "Sunshine": ("Sunshine", "hours", "Approximate hours of bright sunshine."),
    "WindGustDir": ("Wind gust direction", "code", "Direction code used in the original dataset."),
    "WindGustspeed": ("Wind gust speed", "km/h", "Strongest wind gust speed."),
    "WindDir9am": ("Wind direction at 9 AM", "code", "Direction code used in the original dataset."),
    "WindDir3pm": ("Wind direction at 3 PM", "code", "Direction code used in the original dataset."),
    "WindSpeed9am": ("Wind speed at 9 AM", "km/h", "Wind speed measured at 9 AM."),
    "WindSpeed3pm": ("Wind speed at 3 PM", "km/h", "Wind speed measured at 3 PM."),
    "Humidity9am": ("Humidity at 9 AM", "%", "Moisture in the air at 9 AM."),
    "Humidity3pm": ("Humidity at 3 PM", "%", "Moisture in the air at 3 PM."),
    "Pressure9am": ("Pressure at 9 AM", "hPa", "Air pressure measured at 9 AM."),
    "Pressure3pm": ("Pressure at 3 PM", "hPa", "Air pressure measured at 3 PM."),
    "Cloud9am": ("Cloud cover at 9 AM", "oktas (dataset scale)", "Cloud amount recorded at 9 AM."),
    "Cloud3pm": ("Cloud cover at 3 PM", "oktas (dataset scale)", "Cloud amount recorded at 3 PM."),
    "Temp9am": ("Temperature at 9 AM", "°C", "Temperature measured at 9 AM."),
    "Temp3pm": ("Temperature at 3 PM", "°C", "Temperature measured at 3 PM."),
    "RainToday": ("Did it rain today?", "", "Choose Yes or No."),
}


def build_form_fields():
    fields = []
    for name in FEATURE_NAMES:
        series = DATA[name]
        label, unit, help_text = FEATURE_INFO.get(name, (name, "", "Enter the weather value."))
        if name in DIRECTION_FEATURES:
            fields.append({"name": name, "label": label, "unit": unit, "kind": "select",
                           "options": sorted(int(value) for value in series.unique()),
                           "default": int(series.median()), "help": help_text})
        elif name == "RainToday":
            fields.append({"name": name, "label": label, "unit": unit, "kind": "select",
                           "options": [0, 1], "default": int(series.mode().iloc[0]), "help": help_text})
        else:
            fields.append({"name": name, "label": label, "unit": unit, "kind": "number",
                           "default": round(float(series.median()), 2),
                           "min": float(series.min()), "max": float(series.max()),
                           "step": "0.1", "help": help_text})
    return fields


FORM_FIELDS = build_form_fields()
FIELD_BY_NAME = {field["name"]: field for field in FORM_FIELDS}


def create_eda_charts():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    counts = DATA[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(["No Rain (0)", "Rain (1)"], counts.values)
    ax.set_title("Rain / No Rain records")
    ax.set_ylabel("Number of records")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, fmt="%d", padding=3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "rain_classes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    rainfall = DATA["Rainfall"].clip(upper=DATA["Rainfall"].quantile(0.99))
    ax.hist(rainfall, bins=24, edgecolor="white")
    ax.set_title("Daily rainfall distribution (up to 99th percentile)")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Number of records")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "rainfall_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for label, group in DATA.groupby(TARGET):
        ax.hist(group["Humidity3pm"], bins=18, alpha=0.55, label="Rain" if label == 1 else "No Rain")
    ax.set_title("Afternoon humidity by tomorrow's result")
    ax.set_xlabel("Humidity at 3 PM (%)")
    ax.set_ylabel("Number of records")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "humidity_by_rain.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    correlations = DATA.corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(8).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(correlations.index, correlations.values)
    ax.set_title("Features most related to RainTomorrow")
    ax.set_xlabel("Absolute correlation")
    ax.set_xlim(0, max(0.1, float(correlations.max()) * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


create_eda_charts()


def get_sample_examples():
    """Find real rows the current model classifies correctly for demo buttons."""
    examples = {}
    sample_data = DATA.sample(n=min(1200, len(DATA)), random_state=7)
    feature_values = sample_data[FEATURE_NAMES].to_numpy(dtype=float)
    predictions = MODEL.predict(feature_values)
    for expected in (0, 1):
        matches = np.flatnonzero((predictions == expected) & (sample_data[TARGET].to_numpy() == expected))
        if len(matches):
            row = sample_data.iloc[int(matches[0])]
            examples[str(expected)] = {name: float(row[name]) for name in FEATURE_NAMES}
    return examples


SAMPLE_EXAMPLES = get_sample_examples()


def get_preview():
    return DATA.head(6).round(2).to_dict(orient="records")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_values = {field["name"]: field["default"] for field in FORM_FIELDS}
    if request.method == "POST":
        try:
            values = []
            for name in FEATURE_NAMES:
                field = FIELD_BY_NAME[name]
                raw = request.form.get(name, "").strip()
                if not raw:
                    raise ValueError(f"Please enter {field['label']}")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"{field['label']} must be a valid number.")
                if field["kind"] == "select" and value not in field["options"]:
                    raise ValueError(f"Please choose a valid option for {field['label']}.")
                if field["kind"] == "number" and not field["min"] <= value <= field["max"]:
                    raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']} (dataset range).")
                values.append(value)
                input_values[name] = raw
            result = int(MODEL.predict(np.asarray(values, dtype=float).reshape(1, -1))[0])
            prediction = {"rain": result == 1, "code": result,
                          "label": "Rain may occur tomorrow" if result == 1 else "Rain is not expected tomorrow"}
        except (ValueError, TypeError) as error:
            flash(str(error), "error")

    counts = DATA[TARGET].value_counts().to_dict()
    total = len(DATA)
    no_rain = int(counts.get(0, 0))
    rain = int(counts.get(1, 0))
    missing = int(DATA.isna().sum().sum())
    return render_template(
        "index.html", student_name="Uday Sai Kandregula", roll_number="Y26MC20032",
        university="Acharya Nagarjuna University", course="Master of Computer Applications (MCA)",
        total_records=total, total_features=len(FEATURE_NAMES), no_rain=no_rain, rain=rain,
        no_rain_pct=no_rain / total * 100, rain_pct=rain / total * 100,
        missing_values=missing, feature_fields=FORM_FIELDS, basic_fields=[FIELD_BY_NAME[n] for n in BASIC_FEATURES],
        advanced_fields=[FIELD_BY_NAME[n] for n in FEATURE_NAMES if n not in BASIC_FEATURES],
        input_values=input_values, prediction=prediction, preview=get_preview(),
        sample_examples=SAMPLE_EXAMPLES, model_results=MODEL_RESULTS,
        dataset_name=DATA_PATH.name, feature_names=FEATURE_NAMES,
        top_correlations=DATA.corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(5).items(),
    )


@app.route("/health")
def health():
    return {"status": "ok", "dataset": DATA_PATH.name, "records": len(DATA), "model_loaded": MODEL is not None}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
