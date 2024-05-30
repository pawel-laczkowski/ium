import torch
import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from NeuralNetwork import NeuralNetwork

# Load model if it exists
if os.path.exists('./models/model.pth'):
    # Create model
    model = torch.load('./models/model.pth')

    # Load test data
    test = pd.read_csv('./datasets/test.csv')

    # Split data
    X_test = test.drop(columns=['id', 'diagnosis']).values
    y_test = test['diagnosis'].values

    # Convert data to PyTorch tensors
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # Predict
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
else:
    raise FileNotFoundError('Model not found')
