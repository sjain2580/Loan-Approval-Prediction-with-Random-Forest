# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE

# Step 2: Load the dataset
data_url = 'loan_prediction.csv'
try:
    df = pd.read_csv(data_url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset from URL: {e}")
    print("Please ensure you have an active internet connection.")
    exit()

# Step 3: Exploratory Data Analysis (EDA) - A key step for professional portfolios
# EDA helps us understand the data, identify patterns, and prepare for modeling.
print("\nStarting Exploratory Data Analysis (EDA)...")

# Visualize the distribution of the target variable 'Loan_Status'
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Distribution of Loan Status')
plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
plt.ylabel('Count')
plt.savefig('distribution.png')
plt.show()

# Visualize the relationship between categorical features and Loan Status
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, col in enumerate(categorical_cols):
    row, ax_col = i // 3, i % 3
    sns.countplot(x=col, hue='Loan_Status', data=df, ax=axes[row, ax_col])
    axes[row, ax_col].set_title(f'Loan Status by {col}')
plt.tight_layout()
plt.savefig('Categorical_features_vs_Loan_status.png')
plt.show()

# Visualize the correlation matrix of numerical features
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation.png')
plt.show()

print("EDA complete. Insights gained from visualizations will guide our modeling decisions.")

# Step 4: Data Preprocessing
print("\nStarting data preprocessing...")

# Clean up column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()

# Drop the unique identifier column 'Loan_ID' as it is not useful for the model.
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)
else:
    print("Warning: 'Loan_ID' column not found. Skipping drop.")

# Handle missing values.
# For categorical columns, we fill missing values with the mode (most frequent value).
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
# For numerical columns, we fill missing values with the mean.
for col in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Convert categorical features into numerical format.
# Label Encoding for binary features
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
# Explicitly convert 'Dependents' to string before replacing and converting to float
df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False).astype(float)
df['Education'] = le.fit_transform(df['Education'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Property_Area'] = le.fit_transform(df['Property_Area'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

print("Data preprocessing complete.")
print(df.head())
print("\nMissing values after preprocessing:\n", df.isnull().sum())

# Step 5: Split the data into features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the data into a training set and a testing set.
# We use 'stratify=y' to ensure the train/test split has a similar proportion of each class.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nData split into training and testing sets.")
print(f"Training data shape before SMOTE: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Use SMOTE to handle imbalanced data
print("\nApplying SMOTE to the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("SMOTE applied successfully.")
print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts())

# Step 6: Hyperparameter Tuning
# We'll use GridSearchCV to find the best parameters for the Random Forest model.
param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees in the forest
    'max_depth': [5, 10, 15, None],    # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required at each leaf node
}

# Create a Random Forest Classifier instance.
rf_classifier = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best combination of parameters.
print("\nStarting hyperparameter tuning with GridSearchCV. This may take a few moments...")
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# The best estimator is the model with the best parameters.
best_model = grid_search.best_estimator_

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best cross-validation F1-Score: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)
print("-------------------------------------")

# Step 7: Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate and print evaluation metrics.
print("\n--- Final Model Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("------------------------------")

# Step 8: Visualization - Feature Importance Plot
# Get feature importances from the best model.
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

# Plot the feature importances.
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance.png')
plt.show()

# Step 9: Save the trained model for later use
joblib.dump(best_model, 'loan_approval_model.pkl')
print("\nModel saved as 'loan_approval_model.pkl'")
