# AI-Stock-Prediction
# A time-series forecasting model built with LSTM neural networks in PyTorch to predict future stock prices based on historical closing data. The project processes CSV input, trains an LSTM model, and visualizes predicted vs. actual prices for trend analysis. Used PyTorch, LSTM, pandas, matplotlib, MinMaxScaler.
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Load and Prepare Data ---
def load_data(filename, sequence_length=50):
    df = pd.read_csv(filename)
    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(prices) - sequence_length):
        X.append(prices[i:i + sequence_length])
        y.append(prices[i + sequence_length])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, scaler

# --- Define LSTM Model ---
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Training Function ---
def train_model(model, X, y, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# --- Plot Predictions ---
def plot_predictions(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        predicted = model(X).numpy()
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y.numpy())

    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    filename = "stock_data.csv"  # Replace with your actual CSV file
    X, y, scaler = load_data(filename)

    model = StockLSTM().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train_model(model, X, y)
    plot_predictions(model, X, y, scaler)
