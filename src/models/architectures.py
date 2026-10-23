"""Shared neural architecture for training and inference, checkpoint compatible."""
import torch
from torch import nn

from src.core.config import KPIS

KPI_FEATURES = len(KPIS)  # input and output channels; checkpoint shapes stay the same

class ForecastNet(nn.Module):

    def __init__(
        self,
        mode,
        seq_len,
        hidden=32,
    ):

        super().__init__()

        self.mode = mode

        if mode == "lstm":

            self.sequence = nn.LSTM(
                input_size=KPI_FEATURES,
                hidden_size=hidden,
                batch_first=True,
            )

        elif mode == "gru":

            self.sequence = nn.GRU(
                input_size=4,
                hidden_size=hidden,
                batch_first=True,
            )

        elif mode == "transformer":

            self.project = nn.Linear(KPI_FEATURES, hidden)

            self.position = nn.Parameter(
                torch.zeros(1, seq_len, hidden)
            )

            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=False,
            )

            self.sequence = nn.TransformerEncoder(
                layer,
                num_layers=1,
                enable_nested_tensor=False,
            )

        else:
            raise ValueError(
                f"Unknown model: {mode}"
            )

        # Four KPI outputs.
        # Add seasonal information to the final prediction head.
        self.head = nn.Sequential(
            nn.Linear(hidden + KPI_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, KPI_FEATURES),
        )

    def forward(self, seq, seasonal):

        if self.mode == "transformer":

            embedded = (
                self.project(seq)
                + self.position
            )

            hidden = self.sequence(
                embedded
            )[:, -1, :]

        else:

            hidden = self.sequence(
                seq
            )[0][:, -1, :]

        combined = torch.cat(
            [hidden, seasonal],
            dim=-1,
        )

        return self.head(combined)
