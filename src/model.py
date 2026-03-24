import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        """
        Binary Classification LSTM:
        - Predicts the PROBABILITY of a price jump (UP > 0.5, DOWN < 0.5).
        - Includes Sigmoid layer for normalized class probability.
        """
        super(LSTMModel, self).__init__()
        # Using dropout for robustness
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim) # Output for probability
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        
        # Take the last time step for forecasting the next jump
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.fc(out)
        
        return self.sigmoid(out) # Probability [0, 1]
