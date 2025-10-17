import pickle
import json
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load data
print("Loading data...")
X = pd.read_csv('data/raw/X.csv')
y = pd.read_csv('data/raw/Y.csv')

print(f"Data shape - X: {X.shape}, y: {y.shape}")

# Split data
test_size = params['train']['test_size']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(
    max_iter=params['train']['epochs'] * 10,
    random_state=42
)
model.fit(X_train_scaled, y_train.values.ravel())

# Calculate training accuracy
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to models/model.pkl")

# Save metrics
metrics = {
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('metrics/train_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Training completed successfully!")
