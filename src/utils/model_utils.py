import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def save_model(model, filepath):
    torch.save(model, filepath)

def load_model(filepath):
    return torch.load(filepath)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

