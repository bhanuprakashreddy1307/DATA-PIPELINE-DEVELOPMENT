import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Step 1: Load Data
data = pd.DataFrame({
    'age': [25, 30, 35, None, 40],
    'salary': [50000, 60000, 80000, 100000, None],
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'purchased': [0, 1, 0, 1, 0]
})

# Step 2: Separate features and target variable
X = data.drop(columns=['purchased'])
y = data['purchased']

# Step 3: Define preprocessing for numerical and categorical features
numeric_features = ['age', 'salary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Step 4: Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build a pipeline with preprocessing and model training
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 7: Train the model
model_pipeline.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Step 9: Make predictions on new data
new_data = pd.DataFrame({
    'age': [29],
    'salary': [70000],
    'gender': ['female']
})
predictions = model_pipeline.predict(new_data)
print(f'Prediction for new data: {predictions[0]}')

