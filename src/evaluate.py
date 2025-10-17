import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading model...")
# Load model
with open('models/model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']

print("Loading data...")
# Load data
X = pd.read_csv('data/raw/X.csv')
y = pd.read_csv('data/raw/Y.csv')

# Preprocess
X_scaled = scaler.transform(X)

# Make predictions
print("Making predictions...")
predictions = model.predict(X_scaled)

# Calculate metrics
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average='weighted', zero_division=0)
recall = recall_score(y, predictions, average='weighted', zero_division=0)
f1 = f1_score(y, predictions, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save evaluation metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'total_samples': len(X)
}

with open('metrics/eval_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed successfully!")
