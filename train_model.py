# ==========================================
# AI Student Performance Prediction System
# Model Training
# ==========================================

# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------------------
# Input and Output
# ------------------------------------------

X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# Random Forest Model
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------
# Evaluation
# ------------------------------------------

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nModel Performance")
print("----------------------")
print(f"R2 Score : {r2:.3f}")
print(f"MAE      : {mae:.2f}")

# ------------------------------------------
# Save Model
# ------------------------------------------

joblib.dump(model, "random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel Saved Successfully!")