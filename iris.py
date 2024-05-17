import csv
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load CSV data
with open('iris.csv', 'r') as file:
    csv_data = csv.reader(file)
    header = next(csv_data)
    data = [row for row in csv_data]

# Step 2: Connect to SQLite3 database (or create it)
conn = sqlite3.connect('iris_data.db')
cursor = conn.cursor()

# Step 3: Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Iris (
        sepal_length REAL,
        sepal_width REAL,
        petal_length REAL,
        petal_width REAL,
        species TEXT
    )
''')

# Step 4: Insert data into table
cursor.executemany('''
    INSERT INTO Iris (sepal_length, sepal_width, petal_length, petal_width, species) 
    VALUES (?, ?, ?, ?, ?)
''', data)

# Step 5: Commit the changes
conn.commit()

# Perform a self-join to compare rows within the same table
query = '''
    SELECT 
        a.sepal_length AS sepal_length_a, 
        a.sepal_width AS sepal_width_a, 
        a.petal_length AS petal_length_a, 
        a.petal_width AS petal_width_a, 
        a.species AS species_a,
        b.sepal_length AS sepal_length_b, 
        b.sepal_width AS sepal_width_b, 
        b.petal_length AS petal_length_b, 
        b.petal_width AS petal_width_b, 
        b.species AS species_b
    FROM Iris a
    JOIN Iris b ON a.species = b.species AND a.rowid != b.rowid
'''

# Fetch data and load into DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Display DataFrame to verify
print(df.head())

# Assuming the target column is 'species_a'
X = df.drop(columns=['species_a', 'species_b'])
y = df['species_a']

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Perform profiling on train data
profile = ProfileReport(pd.concat([X_train, y_train], axis=1))
profile.to_file("train_data_profile.html")

# Categorize data into categorical and numerical values
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Identify null and missing values
null_counts = X_train.isnull().sum()
print(null_counts)

# Correlation heatmap (numeric columns only)
plt.figure(figsize=(10, 8))
sns.heatmap(X_train[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.show()

# Violin plots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=X_train[col], y=y_train)
    plt.show()

# Analyzing correlations
high_corr = X_train[numerical_cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=False)
high_corr = high_corr[high_corr >= 0.5]
print(high_corr)

# Distribution analysis
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(X_train[col], kde=True)
    plt.show()

# Define a class-based preprocessor
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, numerical_cols),
                ('cat', self.cat_transformer, categorical_cols)
            ]
        )
    
    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        return self.preprocessor.transform(X)
