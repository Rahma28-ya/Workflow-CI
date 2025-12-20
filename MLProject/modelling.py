import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Path dataset
DATA_PATH = "namadataset_preprocessing/telco_customer_churn_preprocessing.csv"
# Folder model untuk MLflow
MODEL_DIR = "MLProject/model"

# Pastikan folder model ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Log metric dan model ke MLflow
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, artifact_path=MODEL_DIR)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))