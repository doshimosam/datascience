import sys
import subprocess
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ▶ Ensure required library installed
print("Installing imbalanced-learn if missing...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
except:
    pass
print("✔ Dependencies ok!")


# -------------------------------------------------------
# 1️⃣ LOAD + CLEAN + FEATURE ENGINEERING
# -------------------------------------------------------
def load_and_prepare_data(file_path):
    print(f"\n📌 Loading dataset: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found. Please check the path.")

    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = df.columns.str.strip()

    # Rename columns
    df.rename(columns={
        'DATE': 'time',
        'HOURLYVISIBILITY': 'visibility',
        'HOURLYDRYBULBTEMPF': 'temperature',
        'HOURLYDewPointTempF': 'dew_point',
        'HOURLYRelativeHumidity': 'humidity',
        'HOURLYWindSpeed': 'wind_speed'
    }, inplace=True)

    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Convert numeric columns
    for col in ['visibility', 'temperature', 'dew_point', 'humidity', 'wind_speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    # Target Engineering
    df['fog'] = (df['visibility'] < 1.0).astype(int)
    df['hour'] = df['time'].dt.hour
    df['temp_dew_point_diff'] = abs(df['temperature'] - df['dew_point'])

    print("✔ Data preparation completed!")
    return df


# -------------------------------------------------------
# 2️⃣ MODEL TRAINING + TUNING
# -------------------------------------------------------
def tune_and_train_model(X_train, y_train):
    print("\n🔍 Running Hyperparameter Tuning (GridSearchCV)...")

    model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(model, param_grid, scoring='recall', cv=5,
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n🏆 Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


# -------------------------------------------------------
# 3️⃣ VISUALIZATION FUNCTIONS
# -------------------------------------------------------
def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["No Fog (0)", "Fog (1)"],
                yticklabels=["No Fog (0)", "Fog (1)"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y):
    print("\n📈 Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Accuracy")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Testing Accuracy")
    plt.title("Training vs Testing Accuracy Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# 4️⃣ MAIN PIPELINE
# -------------------------------------------------------
if __name__ == '__main__':
    DATA_PATH = 'D:/fog_prediction/jfk_weather_cleaned.csv'


    df = load_and_prepare_data(DATA_PATH)

    features = ['hour', 'humidity', 'temp_dew_point_diff', 'wind_speed', 'temperature']
    X = df[features]
    y = df['fog']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    print("\n📊 Class Distribution Before SMOTE:")
    print(y_train.value_counts())

    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    print("\n📊 Class Distribution After SMOTE:")
    print(y_train_resampled.value_counts())

    model = tune_and_train_model(X_train_resampled, y_train_resampled)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print("\n🎯 Final Model Performance")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, zero_division=0))

    plot_confusion_matrix(y_test, predictions)
    plot_learning_curve(model, X, y)

    print("\n🚀 Fog Prediction Model Successfully Completed!")
