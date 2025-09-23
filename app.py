import pandas as pd

def load_data(x_path, y_path):
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    return X, Y

if __name__ == '__main__':
    X, Y = load_data('X.csv', 'Y.csv')
    print(f'Loaded X shape: {X.shape}, Y shape: {Y.shape}')