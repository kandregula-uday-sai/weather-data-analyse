# Weather Data Analysis and Rain Prediction

**MCA Mini Project**

This project studies past weather records and predicts whether it may rain tomorrow. It uses Python, Pandas, a Random Forest model written for this project, and a small Flask website.

## Student details

- **Student:** Uday Sai Kandregula
- **Roll number:** Y26MC20032
- **Course:** Master of Computer Applications (MCA)
- **University:** Acharya Nagarjuna University

## 1. What can I do in this project?

1. View a summary of the weather dataset.
2. See EDA charts (simple graphs that help us understand the data).
3. Enter weather details and get a Rain / No Rain result.
4. Try sample weather records from the dataset.
5. View model test scores and a confusion matrix.

**Please note:** The dataset's output column is `RainTomorrow`. So the current model predicts Rain or No Rain. It does not give a temperature number. A separate regression model would be needed for temperature prediction.

## 2. Project files

```text
Weather_Rain_Prediction_MCA/
│
├── app.py                         # Opens the Flask website
├── train.py                       # Trains and tests the model
├── weather_data_shuffled.csv      # Main dataset used by the project
├── model_results.json             # Model test results
├── requirements.txt               # Python packages
├── README.md                      # Project guide
├── start_project.bat              # Start the website on Windows
│
├── models/
│   ├── random_forest.py           # Decision Tree and Random Forest code
│   └── rain_prediction_model.pkl  # Trained model
│
├── templates/
│   ├── index.html                 # Main website page
│   └── field.html                 # Weather input field design
│
└── static/
    ├── style.css                  # Website design
    └── charts/                    # EDA graphs (created by app.py)
```

## 3. Dataset details

The main dataset is `weather_data_shuffled.csv`.

- Rows: **34,850**
- Columns: **21**
- Input features: **20**
- Output column: **RainTomorrow**
- No Rain (`0`): **20,910 records**
- Rain (`1`): **13,940 records**

The records are randomly shuffled. Shuffling changes the order of rows; it does not change the original values or class counts. The dataset is not forced to a 50:50 ratio.

### What are 0 and 1?

- `0` means No Rain tomorrow.
- `1` means Rain tomorrow.

### Main input features

- Temperature: `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`
- Rain and sunshine: `Rainfall`, `RainToday`, `Sunshine`, `Evaporation`
- Humidity and clouds: `Humidity9am`, `Humidity3pm`, `Cloud9am`, `Cloud3pm`
- Wind: `WindGustDir`, `WindGustspeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`
- Air pressure: `Pressure9am`, `Pressure3pm`

The three wind direction columns use numeric codes from the dataset. The original data does not provide a compass-direction lookup, so the website keeps the same codes used during model training.

## 4. Run the project on Windows 10 or 11

### First-time setup

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. During installation, select **Add Python to PATH**.
3. Extract this project ZIP into a normal folder, for example `D:\MCA\Weather_Rain_Prediction_MCA`.
4. Open the folder and double-click `start_project.bat`.
5. Wait for the package installation to finish.
6. Open this address in Chrome or Edge: **http://127.0.0.1:5000**.

Keep the black Command Prompt window open while using the website. Press `Ctrl + C` in that window to stop Flask.

### If you want to use PowerShell commands

Open PowerShell inside the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Then open **http://127.0.0.1:5000**. These commands use the virtual-environment Python directly, so you do not need to activate it or change PowerShell's execution policy.

## 5. How to use the website

1. **Home:** Check the number of records and weather inputs.
2. **Data Analysis:** Look at the four EDA graphs.
3. **Try Prediction:** Enter the six main weather values.
4. If you want, open **More weather details** to edit the other 14 inputs.
5. You can click **Load No Rain example** or **Load Rain example** to fill the form with a real dataset example.
6. Click **Predict Rain Tomorrow**.
7. Read the result: `0 = No Rain` or `1 = Rain`.

The extra 14 fields already have example values from the dataset. You do not have to open that section for a basic test. The model still receives all 20 values.

## 6. What is EDA?

EDA means Exploratory Data Analysis. It is the step where we study the dataset before building a machine-learning model.

The website shows:

- **Rain / No Rain chart:** Counts how many records belong to each class.
- **Rainfall chart:** Shows how rainfall values are distributed.
- **Humidity chart:** Compares afternoon humidity for Rain and No Rain records.
- **Correlation chart:** Shows which numeric features have a stronger relationship with `RainTomorrow`. Correlation does not mean that a feature causes rain.

## 7. How the Random Forest works

1. The dataset is separated into training data (80%) and testing data (20%).
2. Ten decision trees are trained using random samples of the training data.
3. Each tree uses Gini impurity to choose splits.
4. The trees make their own predictions.
5. The Random Forest uses majority voting to give the final result.
6. The model is tested using accuracy, precision, recall, F1 score and a confusion matrix.

The Random Forest code is written in `models/random_forest.py`. Scikit-learn is used for splitting the dataset and calculating evaluation scores; the model itself is our custom implementation.

## 8. Model results

The model included in this project was trained using `weather_data_shuffled.csv` with an 80:20 train/test split and `random_state=42`.

| Measure | Result from this training run |
|---|---:|
| Accuracy | 81.91% |
| Precision | 80.85% |
| Recall | 71.77% |
| F1 score | 76.04% |

These are results from one train/test split. They are not a promise that every new prediction will be correct.

## 9. Train the model again (optional)

The trained model is already included, so you can open the website without training again.

If you change the dataset or want to repeat training, open Command Prompt in the project folder and run:

```bat
.venv\Scripts\python.exe train.py
```

Or, after first-time setup, double-click `train_model.bat`. Training replaces `models/rain_prediction_model.pkl` and updates `model_results.json`.

## 10. Common problems

**Python is not recognised**

Install Python again and select **Add Python to PATH**. Close and reopen the terminal.

**A package is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**The model file is missing**

Run `train.py` once to create it.

**Port 5000 is already in use**

Close the other program using port 5000, then start this project again.

---

**Project note:** This is an educational demonstration using historical weather records. It is not connected to a live weather station or official forecast service.
