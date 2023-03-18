#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openpyxl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))
        #x = torch.softmax(self.fc5(x), dim=1)
        return x
    
    def train(self, train_data, train_labels, epochs=1000, batch_size=8192, lr=0.05):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def predict(self, test_data):
        outputs = self.forward(test_data)
        _, y_pred = torch.max(outputs, dim=1)
        return y_pred.numpy()
    
# Load the data from Excel
wb = openpyxl.load_workbook("./data/university_rank/rank_processed.xlsx")
sheet = wb.active
X, y = [], []
for row in sheet.iter_rows(min_row=2, values_only=True):
    X.append([row[2], row[7], row[8],row[9]])  # C、H、I and J
    y.append(row[10])   # K
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

print_every = 5
# Create and train the neural network
nn = NeuralNetwork(input_size=4, output_size=2)
nn.train(X_train, y_train, epochs=15)


# Make predictions on the testing set and calculate accuracy
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test.numpy())
print(f"Accuracy: {accuracy}")


# In[ ]:




