from preprocess import load_data, preprocess_data
from utils import evaluate_model, plot_metrics_comparison
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# Load and preprocess data
file_path = r'C:\Users\stef\Documents\Uni\EECE\EECE 490\assignment 3\Breast Cancer Classification\data\breast_cancer_data.csv'
X, y = load_data(file_path)
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# List to hold metrics for all models
metrics_list = []

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
metrics_list.append(evaluate_model(y_test, y_pred_log, "Logistic Regression"))

# Support Vector Machine Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
metrics_list.append(evaluate_model(y_test, y_pred_svm, "SVM"))

# Feed-Forward Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Initialize and train the neural network
input_size = X_train.shape[1]
nn_model = NeuralNetwork(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):  # Adjust epochs as needed
    nn_model.train()
    optimizer.zero_grad()
    outputs = nn_model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()

# Evaluate Neural Network
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(torch.tensor(X_test, dtype=torch.float32))
    y_pred_nn = torch.round(outputs).squeeze().numpy()
metrics_list.append(evaluate_model(y_test, y_pred_nn, "Neural Network"))

# Plot the metrics comparison
plot_metrics_comparison(metrics_list)
