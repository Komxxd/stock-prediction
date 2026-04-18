import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        """
        Binary Classification LSTM:
        - Predicts the Logits of price direction.
        - Using BCEWithLogitsLoss for numerical stability.
        """
        super(LSTMModel, self).__init__()
        # PyTorch only applies dropout between layers, so it requires num_layers > 1
        d_rate = 0.3 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=d_rate)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        
        # Take the last time step
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.fc(out)
        
        return out # Returns Logits
