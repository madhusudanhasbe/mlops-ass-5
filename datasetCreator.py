from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=10,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

np.savetxt('X.csv', X, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')