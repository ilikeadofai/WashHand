import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_and_evaluate(X_path, y_path, model_path):
    X = np.load(X_path)
    y = np.load(y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate handwashing gesture classifier.')
    parser.add_argument('--x', default='X.npy', help='Path to X.npy')
    parser.add_argument('--y', default='y.npy', help='Path to y.npy')
    parser.add_argument('--model', default='models/handwash_model.pkl', help='Path to save trained model')
    args = parser.parse_args()
    train_and_evaluate(args.x, args.y, args.model) 