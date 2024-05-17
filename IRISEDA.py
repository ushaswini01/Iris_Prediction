import csv
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import joblib
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='punnaushaswini', repo_name='Iris_prediction', mlflow=True)

# Load data from CSV and insert into SQLite database
with open('iris.csv', 'r') as file:
    csv_data = csv.reader(file)
    header = next(csv_data)
    data = [row for row in csv_data]

conn = sqlite3.connect('iris_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Iris (
        sepal_length REAL,
        sepal_width REAL,
        petal_length REAL,
        petal_width REAL,
        species TEXT
    )
''')
cursor.executemany('''
    INSERT INTO Iris (sepal_length, sepal_width, petal_length, petal_width, species) 
    VALUES (?, ?, ?, ?, ?)
''', data)
conn.commit()
conn.close()

# Load data from SQLite database
conn = sqlite3.connect('iris_data.db')
query = '''
    SELECT 
        sepal_length, 
        sepal_width, 
        petal_length, 
        petal_width, 
        species
    FROM Iris
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Ensure all numerical columns contain only numeric values
df['sepal_length'] = pd.to_numeric(df['sepal_length'], errors='coerce')
df['sepal_width'] = pd.to_numeric(df['sepal_width'], errors='coerce')
df['petal_length'] = pd.to_numeric(df['petal_length'], errors='coerce')
df['petal_width'] = pd.to_numeric(df['petal_width'], errors='coerce')

X = df.drop(columns=['species'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Generate EDA report
profile = ProfileReport(pd.concat([X_train, y_train], axis=1), title="Iris Dataset EDA Report", explorative=True)
profile.to_file("iris_eda_report.html")

# Define preprocessing pipelines for numerical and categorical features
numerical_cols = X_train.select_dtypes(include=['number']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ]
)

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])

# Set up MLFlow
mlflow.set_tracking_uri(uri = "http://127.0.0.1:8080")  # Add this line to set the tracking URI
mlflow.set_experiment("Iris_Experiment")

# Turn on autologging
mlflow.autolog()

# Train the model and log metrics with MLFlow
with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(model, "random_forest_model")

# Save the model
joblib.dump(model, "model.joblib")
