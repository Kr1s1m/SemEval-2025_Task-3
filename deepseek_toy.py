import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Configuration
CONFIG = {
    "feature_dim": 12,  # Number of features per token
    "hidden_dim": 64,  # LSTM hidden size
    "num_layers": 2,  # Number of LSTM layers
    "dropout": 0.3,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "seq_length": 64  # Max sequence length
}


class HallucinationDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                features = self.extract_features(data)
                self.samples.append(features)

        # Pad sequences and create masks
        self.padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in self.samples],
            batch_first=True,
            padding_value=0
        )

    def extract_features(self, data):
        """Create time-series features for each token"""
        logits = np.array(data['model_output_logits'])
        tokens = data['model_output_tokens']

        features = []
        for i in range(len(tokens)):
            # Base features
            window_size = 3
            context = logits[max(0, i - window_size):i + window_size + 1]

            feat = [
                logits[i],  # Current logit
                np.mean(context),  # Moving average
                np.min(context),  # Local minimum
                np.max(context) - np.min(context),  # Local volatility
                i / len(logits),  # Positional encoding
                len(tokens[i]),  # Token length
                int(bool(tokens[i].strip().isalnum())),  # Is alphanumeric
                np.percentile(logits, 25),  # Global 25th percentile
                np.percentile(logits, 50),  # Global median
                np.percentile(logits, 75),  # Global 75th percentile
                logits[i] - np.median(logits),  # Deviation from median
                np.diff(logits).mean() if i > 0 else 0,  # Trend
            ]

            features.append(feat)

        return np.array(features)


class HallucinationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config["feature_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=True,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=config["hidden_dim"] * 1,
            hidden_size=config["hidden_dim"],
            num_layers=1,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_dim"],
            num_heads=4,
            dropout=config["dropout"]
        )

        self.output_layer = nn.Linear(config["hidden_dim"], config["feature_dim"])

    def forward(self, x):
        # Encoder
        enc_out, (h_n, c_n) = self.encoder(x)

        # Attention
        attn_out, _ = self.attention(
            enc_out.transpose(0, 1),
            enc_out.transpose(0, 1),
            enc_out.transpose(0, 1)
        )

        # Decoder
        dec_out, _ = self.decoder(attn_out.transpose(0, 1))

        # Reconstruction
        reconstructed = self.output_layer(dec_out)
        return reconstructed


# Training Pipeline
def train_model(file_path):
    dataset = HallucinationDataset(file_path)
    dataloader = DataLoader(dataset.padded,
                            batch_size=CONFIG["batch_size"],
                            shuffle=True)

    model = HallucinationModel(CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            # Mask padding
            mask = (batch != 0).any(dim=-1).float()
            reconstructed = model(batch.float())

            loss = (criterion(reconstructed, batch.float()) * mask.unsqueeze(-1)).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    return model


# Inference and Span Detection
def detect_hallucinations(model, sample_data):
    features = HallucinationDataset.extract_features(sample_data)
    with torch.no_grad():
        reconstructed = model(torch.tensor(features).unsqueeze(0).float())

    # Calculate reconstruction error
    error = torch.abs(torch.tensor(features) - reconstructed.squeeze(0)).mean(dim=-1)

    # Dynamic thresholding
    threshold = error.mean() + 2 * error.std()
    mask = error > threshold

    # Group consecutive tokens
    spans = []
    current_span = None
    for i, val in enumerate(mask.numpy()):
        if val:
            if current_span is None:
                current_span = {'start': i, 'end': i}
            else:
                current_span['end'] = i
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        spans.append(current_span)

    return spans


# Usage
model = train_model("data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl")
sample_data =  pd.read_json("data_sets/test_unlabeled/mushroom.en-tst.v1.jsonl", lines=True) # Your input sample
spans = detect_hallucinations(model, sample_data[0])