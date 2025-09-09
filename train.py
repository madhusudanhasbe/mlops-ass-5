import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data
X = np.loadtxt('X.csv', delimiter=',')
y = np.loadtxt('y.csv', delimiter=',')

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training complete and model saved as model.pkl")