import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration
CONFIG = {
    "feature_dim": 12,  # Number of features per token
    "hidden_dim": 64,  # LSTM hidden size
    "num_layers": 2,  # Number of LSTM layers
    "dropout": 0.3,
    "batch_size": 32,
    "num_epochs": 33,
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

    @staticmethod
    def extract_features(data):  # Now properly static
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

    def _extract_features(self, data):
        """Internal wrapper for static method"""
        return self.extract_features(data)


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
            input_size=config["hidden_dim"] * 2,
            hidden_size=config["hidden_dim"],
            num_layers=1,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_dim"] * 2,
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
    # Extract features and get model prediction
    features = HallucinationDataset.extract_features(sample_data)
    with torch.no_grad():
        reconstructed = model(torch.tensor(features).unsqueeze(0).float())

    # Calculate reconstruction error
    error = torch.abs(torch.tensor(features) - reconstructed.squeeze(0)).mean(dim=-1)

    # Dynamic threshold calculation
    threshold = error.mean() + 2 * error.std()

    # Convert errors to probabilities using sigmoid
    probabilities = torch.sigmoid((error - threshold) * 2.0)  # Scale factor 2.0

    # Identify spans with consecutive high probabilities
    spans = []
    current_span = None
    for i in range(len(probabilities)):
        if probabilities[i] > 0.5:  # Probability threshold
            if current_span is None:
                current_span = {
                    'start': i,
                    'end': i,
                    'prob_sum': probabilities[i].item(),
                    'token_indices': [i]
                }
            else:
                current_span['end'] = i
                current_span['prob_sum'] += probabilities[i].item()
                current_span['token_indices'].append(i)
        else:
            if current_span is not None:
                # Calculate span statistics
                span_prob = current_span['prob_sum'] / len(current_span['token_indices'])
                spans.append({
                    'input_text': sample_data['model_input'],
                    'output_text': sample_data['model_output_text'],
                    'start_token_idx': current_span['start'],
                    'end_token_idx': current_span['end'],
                    'probability': round(span_prob, 4),
                    'tokens': [sample_data['model_output_tokens'][idx]
                               for idx in current_span['token_indices']],
                    'mean_error': error[current_span['token_indices']].mean().item(),
                    'token_count': len(current_span['token_indices'])
                })
                current_span = None

    # Add final span if exists
    if current_span is not None:
        span_prob = current_span['prob_sum'] / len(current_span['token_indices'])
        spans.append({
            'input_text': sample_data['model_input'],
            'output_text': sample_data['model_output_text'],
            'start_token_idx': current_span['start'],
            'end_token_idx': current_span['end'],
            'probability': round(span_prob, 4),
            'tokens': [sample_data['model_output_tokens'][idx]
                       for idx in current_span['token_indices']],
            'mean_error': error[current_span['token_indices']].mean().item(),
            'token_count': len(current_span['token_indices'])
        })

    return spans

def read_jsonl(file_path):
    objects = []
    with open(file_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            objects.append(obj)
    return objects

# Usage example
file_path = 'data_sets/test_unlabeled/mushroom.en-tst.v1.jsonl'
file_path = 'data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl'
parsed_objects = read_jsonl(file_path)
# print(parsed_objects)
# Usage
model = train_model("data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl")
sample_data = parsed_objects

size = len(sample_data)
spans = []
# Loop over each record in sample_data
for d in sample_data:
    print(detect_hallucinations(model, d))
    spans.append(detect_hallucinations(model, d))
#print(spans)