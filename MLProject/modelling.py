import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset hasil preprocessing
DATA_PATH = "namadataset_preprocessing/telco_customer_churn_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Target diasumsikan kolom terakhir
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Set experiment MLflow
mlflow.set_experiment("Telco_Customer_Churn")

# Aktifkan autolog
mlflow.sklearn.autolog()

# 4. Training model
with mlflow.start_run(run_name="LogisticRegression_Basic"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
