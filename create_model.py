import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from NeuralNetwork import NeuralNetwork

# Seed for reproducibility
torch.manual_seed(1234)

# Load data
train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')

# Split data
X_train = train.drop(columns=['id', 'diagnosis']).values
y_train = train['diagnosis'].values

X_test = test.drop(columns=['id', 'diagnosis']).values
y_test = test['diagnosis'].values

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Default parameters
hidden_size = 128

# Default learning parameters
learning_rate = 0.001
weight_decay = 0.001
num_epochs = 1000

# Overwrite parameters from arguments
if len(sys.argv) == 5:
    hidden_size = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    weight_decay = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

print(f'Hidden size: {hidden_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, Num epochs: {num_epochs}')

# Parameters
input_size = X_train.shape[1]

# Model initialization
model = NeuralNetwork(input_size, hidden_size)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
model.train()

for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)

    # Compute loss
    loss = criterion(outputs, y_train)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
model.eval()

with torch.no_grad():

    # Make predictions
    y_pred = model(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')


# If directory models does not exist, create it
if not os.path.exists('./models'):
    os.makedirs('./models')

# Save the model
torch.save(model, './models/model.pth')