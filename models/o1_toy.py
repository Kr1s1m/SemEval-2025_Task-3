#!/usr/bin/env python3

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-mpnet-base-v2")


# -----------------------------
# 1. DUMMY EMBEDDING FUNCTIONS
# -----------------------------
# In a real system, you might load a sentence transformer or a pretrained LM
# to embed the input and output tokens. Here we just create random vectors
# to illustrate the concept.

def embed_input_text(text, embed_dim=8):
    """
    Returns a random embedding for the input text.
    Real-world scenario: use a pretrained model or any embedding approach.
    """
    # For demonstration, we seed by the text hash so it's consistent per example
    encoded = model.encode(text)
    return torch.from_numpy(encoded)


def embed_output_token(token, embed_dim=8):
    """
    Returns a random embedding for a single token in the output.
    Real-world scenario: use a pretrained model's token embedding, etc.
    """
    # We'll do the same seeded approach for consistency
    encoded = model.encode(token)
    return torch.from_numpy(encoded)


# -------------------------------------------------------------
# 2. GATING NETWORK (maps log prob -> gating probability [0..1])
# -------------------------------------------------------------
class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # w and b are scalar learnable parameters
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, log_probs):
        """
        log_probs: (num_tokens,) float tensor
        Returns gating_probs: (num_tokens,) in [0,1].
        gating_probs[i] = sigma(w * log_probs[i] + b)
        """
        return torch.sigmoid(self.w * log_probs + self.b)


# -------------------------------
# 3. MAIN TRAINING LOOP (TOY)
# -------------------------------
def train_soft_gating(jsonl_path, embed_dim=8, num_epochs=1):
    """
    Reads data from jsonl_path, trains a gating network for soft masking.
    Minimizes the L2 distance between the 'masked output embedding' and
    the 'input text embedding' as a toy 'consistency loss'.
    """
    gating_net = GatingNetwork()
    optimizer = optim.Adam(gating_net.parameters(), lr=0.01)

    # We'll store the dataset in memory for simplicity
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Extract needed fields
            model_input = data.get("model_input", "")
            output_tokens = data.get("model_output_tokens", [])
            output_logits = data.get("model_output_logits", [])
            # Skip if invalid
            if not output_tokens or not output_logits:
                continue
            dataset.append((model_input, output_tokens, output_logits))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        for (model_input, output_tokens, output_logits) in dataset:
            # Convert logits to torch tensor
            log_probs = torch.tensor(output_logits, dtype=torch.float)

            # Get gating probabilities
            gating_probs = gating_net(log_probs)  # shape: (num_tokens,)

            # 1) Embed input text (dim=embed_dim)
            input_embed = embed_input_text(model_input, embed_dim=embed_dim)

            # 2) Embed each token and apply "soft mask"
            #    (1 - gating_prob) means keep the token if gating_prob is small.
            masked_output_embed = torch.zeros(embed_dim)
            for i, token in enumerate(output_tokens):
                token_embed = embed_output_token(token, embed_dim=embed_dim)
                keep_weight = (1 - gating_probs[i])
                masked_output_embed += keep_weight * token_embed

            # 3) Consistency loss: L2 distance between
            #    masked_output_embed and input_embed
            loss = torch.mean((masked_output_embed - input_embed) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  Avg loss this epoch: {total_loss / len(dataset):.4f}")

    # After training, let's see the gating probabilities we learned
    print("\nTraining complete.\nNow printing gating for each example:")
    gating_net.eval()
    for (model_input, output_tokens, output_logits) in dataset:
        log_probs = torch.tensor(output_logits, dtype=torch.float)
        gating_probs = gating_net(log_probs)  # shape: (num_tokens,)

        # Detach from computation graph and convert to NumPy
        gating_probs_np = gating_probs.detach().cpu().numpy()

        print(f"\nINPUT: {model_input}")
        print("TOKENS and gating probabilities (near 1 => masked out):")
        for token, gp in zip(output_tokens, gating_probs_np):
            print(f"  {token} => gating_prob={gp:.3f}")


##### toy No2


def embed_to_distribution(embedding):
    eps = 1e-8
    exp_emb = torch.exp(embedding)
    sum_exp = torch.sum(exp_emb) + eps
    return exp_emb / sum_exp

def binary_entropy(g):
    # g is gating_probs[i], a value in (0, 1)
    eps = 1e-8
    return - (g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))

def train_with_entropy(
    jsonl_path,
    embed_dim=8,
    num_epochs=3,
    lambda_penalty=0.01,
    lambda_entropy=0.01,
    learning_rate=0.01
):
    """
    Example training loop:
    - Replaces L2 with cross-entropy alignment for consistency.
    - Adds gating entropy penalty for pushing gating to 0 or 1.
    - Optionally, you can also keep the sum(g_i) penalty if you want.
    """
    gating_net = GatingNetwork()
    optimizer = optim.Adam(gating_net.parameters(), learning_rate)

    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            model_input = data.get("model_input", "")
            output_tokens = data.get("model_output_tokens", [])
            output_logits = data.get("model_output_logits", [])
            if not output_tokens or not output_logits:
                continue
            dataset.append((model_input, output_tokens, output_logits))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for (model_input, output_tokens, output_logits) in dataset:
            log_probs = torch.tensor(output_logits, dtype=torch.float)
            gating_probs = gating_net(log_probs)

            # Embed input text => distribution
            input_embed = embed_input_text(model_input, embed_dim=embed_dim)
            input_dist = embed_to_distribution(input_embed)

            # Soft-mask output
            masked_output_embed = torch.zeros(embed_dim)
            for i, token in enumerate(output_tokens):
                token_embed = embed_output_token(token, embed_dim=embed_dim)
                keep_weight = (1 - gating_probs[i])
                masked_output_embed += keep_weight * token_embed

            # Convert masked output embed => distribution
            masked_dist = embed_to_distribution(masked_output_embed)

            # 1) Consistency loss => cross-entropy
            # cross_entropy: - sum( input_dist[i] * log(masked_dist[i]) )
            loss_consistency = -torch.sum(input_dist * torch.log(masked_dist + 1e-8))

            # 2) Gating penalty => sum or mean of gating_probs
            loss_penalty = gating_probs.mean()

            # 3) Gating entropy => push gating to 0 or 1
            gating_entropy = binary_entropy(gating_probs)
            # average across tokens
            gating_entropy_mean = gating_entropy.mean()

            # Combine them
            loss = loss_consistency \
                 + lambda_penalty * loss_penalty \
                 + lambda_entropy * gating_entropy_mean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"  Avg loss: {avg_loss:.4f}")

    print("\nFinished training. Gating probabilities:")
    gating_net.eval()
    for (model_input, output_tokens, output_logits) in dataset:
        log_probs = torch.tensor(output_logits, dtype=torch.float)
        gating_probs = gating_net(log_probs).detach().cpu().numpy()
        print(f"\nINPUT: {model_input}")
        for token, gp in zip(output_tokens, gating_probs):
            print(f"  {token} => gating_prob={gp:.3f}")



if __name__ == "__main__":
    # Example usage
    # Suppose you have 'data.jsonl' in the same folder
    #train_soft_gating(jsonl_path="data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl", embed_dim=768, num_epochs=3)
    train_with_entropy(
        jsonl_path="data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl",
        embed_dim=768,
        num_epochs=24,
        lambda_penalty=1,
        lambda_entropy=1
    )