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

def embed_input_text(text):
    """
    Returns a random embedding for the input text.
    Real-world scenario: use a pretrained model or any embedding approach.
    """
    # For demonstration, we seed by the text hash so it's consistent per example
    encoded = model.encode(text)
    return torch.from_numpy(encoded)


def embed_output_token(token):
    """
    Returns a random embedding for a single token in the output.
    Real-world scenario: use a pretrained model's token embedding, etc.
    """
    # We'll do the same seeded approach for consistency
    encoded = model.encode(token)
    return torch.from_numpy(encoded)


def from_jsonl(jsonl_path):
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            # Extract needed fields
            lang = data.get("lang", "")
            idx = 0
            id = data.get("id", f"train-{lang.lower()}-{idx}")
            model_input = data.get("model_input", "")
            output_tokens = data.get("model_output_tokens", [])
            output_logits = data.get("model_output_logits", [])
            # Skip if invalid
            if not output_tokens or not output_logits:
                continue
            dataset.append((id, model_input, output_tokens, output_logits))
            idx += 1

    return dataset

def labels_from_jsonl(jsonl_path):
    labels = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            # Extract needed fields
            lang = data.get("lang", "")
            idx = 0
            id = data.get("id", f"train-{lang.lower()}-{idx}")
            soft_labels = data.get("soft_labels", [])
            hard_labels = data.get("hard_labels", [])
            # Skip if invalid
            if not soft_labels or not hard_labels:
                continue
            labels.append((id, soft_labels, hard_labels))
            idx += 1

    return labels

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
#def train_soft_gating(data, embed_dim=8, num_epochs=1):
#    """
#    Reads data from data, trains a gating network for soft masking.
#    Minimizes the L2 distance between the 'masked output embedding' and
#    the 'input text embedding' as a toy 'consistency loss'.
#    """
#    gating_net = GatingNetwork()
#    optimizer = optim.Adam(gating_net.parameters(), lr=0.01)
#
#    # We'll store the dataset in memory for simplicity
#    dataset = from_jsonl(data)
#
#   for epoch in range(num_epochs):
#        print(f"Epoch {epoch + 1}/{num_epochs}")
#        total_loss = 0.0
#        for (model_input, output_tokens, output_logits) in dataset:
#            # Convert logits to torch tensor
#            log_probs = torch.tensor(output_logits, dtype=torch.float)
#
#            # Get gating probabilities
#            gating_probs = gating_net(log_probs)  # shape: (num_tokens,)
#
#            # 1) Embed input text (dim=embed_dim)
#            input_embed = embed_input_text(model_input, embed_dim=embed_dim)
#
#            # 2) Embed each token and apply "soft mask"
#            #    (1 - gating_prob) means keep the token if gating_prob is small.
#            masked_output_embed = torch.zeros(embed_dim)
#            for i, token in enumerate(output_tokens):
#                token_embed = embed_output_token(token, embed_dim=embed_dim)
#                keep_weight = (1 - gating_probs[i])
#                masked_output_embed += keep_weight * token_embed
#
#            # 3) Consistency loss: L2 distance between
#            #    masked_output_embed and input_embed
#            loss = torch.mean((masked_output_embed - input_embed) ** 2)
#
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#
#            total_loss += loss.item()
#
#        print(f"  Avg loss this epoch: {total_loss / len(dataset):.4f}")
#
#    # After training, let's see the gating probabilities we learned
#    print("\nTraining complete.\nNow printing gating for each example:")
#    gating_net.eval()
#    for (model_input, output_tokens, output_logits) in dataset:
#        log_probs = torch.tensor(output_logits, dtype=torch.float)
#        gating_probs = gating_net(log_probs)  # shape: (num_tokens,)
#
#        # Detach from computation graph and convert to NumPy
#        gating_probs_np = gating_probs.detach().cpu().numpy()
#
#        print(f"\nINPUT: {model_input}")
#        print("TOKENS and gating probabilities (near 1 => masked out):")
#        for token, gp in zip(output_tokens, gating_probs_np):
#            print(f"  {token} => gating_prob={gp:.3f}")


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


def get_training_results(gating_net, dataset):
    result = []
    gating_net.eval()
    for (id, model_input, output_tokens, output_logits) in dataset:
        log_probs = torch.tensor(output_logits, dtype=torch.float)
        gating_probs = gating_net(log_probs).detach().cpu().numpy()
        result.append((id, output_tokens, gating_probs))
        #print(f"\nINPUT: {model_input}")
        #for token, gp in zip(output_tokens, gating_probs):
            #print(f"  {token} => gating_prob={gp:.3f}")

    print("\nGating probabilities generated.")
    return result


def train_with_entropy(
    data,
    embed_dim=8,
    num_epochs=3,
    lambda_penalty=0.01,
    lambda_entropy=0.01,
    learning_rate=0.01
):
    """
    Training loop:
    - Replaces L2 with cross-entropy alignment for consistency.
    - Adds gating entropy penalty for pushing gating to 0 or 1.
    - Optionally, you can also keep the sum(g_i) penalty if you want.
    """
    gating_net = GatingNetwork()
    optimizer = optim.Adam(gating_net.parameters(), learning_rate)

    dataset = data

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for (id, model_input, output_tokens, output_logits) in dataset:
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

    print("\nFinished training.")
    return gating_net


def spread_gating_probs(
    gating_probs, 
    threshold=0.8, 
    spread_factor=0.1
):
    """
    Given an array of gating_probs in [0,1], for each token i whose gating_probs[i]
    exceeds `threshold`, distribute `spread_factor * gating_probs[i]` to
    its neighbors (i-1, i+1).

    Args:
        gating_probs (np.ndarray): shape (num_tokens,), each in [0,1].
        threshold (float): gating probability above which we spread to neighbors.
        spread_factor (float): fraction of gating_probs[i] to add to neighbors.
          If gating_probs[i] = x and x > threshold, we add (spread_factor * x)
          to the left neighbor, and (spread_factor * x) to the right neighbor,
          and optionally subtract from the current token if we want "mass" conservation.
    
    Returns:
        np.ndarray: new gating probabilities after spreading.
    """

    new_probs = gating_probs.copy()
    n = len(gating_probs)

    for i in range(n):
        prob_i = gating_probs[i]
        if prob_i > threshold:
            # Spread some fraction of prob_i to neighbors
            amount = spread_factor * prob_i
            
            # Left neighbor
            if i > 0:
                new_probs[i - 1] += amount

            # Right neighbor
            if i < n - 1:
                new_probs[i + 1] += amount

            # Optionally, reduce current token's gating prob
            # to "conserve" total gating mass:
            new_probs[i] -= 2 * amount  # because we added to 2 neighbors

            # You might want to ensure it doesn't go below 0:
            new_probs[i] = max(0, new_probs[i])

    # Optionally, you might want to clip values above 1.0
    new_probs = np.clip(new_probs, 0.0, 1.0)

    return new_probs

def spread_all(zipped):
    new_zipped = zipped.copy()
    for (id, tokens, gating_probs) in zipped:
        new_zipped.append((tokens, spread_gating_probs(gating_probs)))

    return new_zipped
    
def generate_spans(zipped):
    spans = []
    for (id, tokens, gating_probs) in zipped:
        i = 0
        span = None
        for token, gating_prob in zip(tokens, gating_probs):
            if gating_prob >= 0.5:
                if span == None:
                    span = {
                        'start': i,
                        'end': i,
                        'prob': gating_prob,
                        'tokens': [token]
                    }
                else:
                    span['end'] = i
                    span['prob'] += gating_prob
                    span['tokens'].append(token)
            else:
                if span is not None:
                    span['prob'] /= (span['end'] - span['start']) + 1
                    spans.append(span)
                    span = None
            i+=1

    return spans

def calculate_error(gating_model, labeled_jsonl_path):
    labeled_dataset = from_jsonl(labeled_jsonl_path)
    zipped = get_training_results(gating_model, labeled_dataset)
    spans = generate_spans(zipped)
    lables = labels_from_jsonl(labeled_jsonl_path)
    for (soft_labels, hard_labels) in lables:


if __name__ == "__main__":
    # Example usage of soft gating
    #train_soft_gating(data="data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl", embed_dim=768, num_epochs=3)

    sources = [
        'data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl',
        'data_sets/test_unlabeled/mushroom.en-tst.v1.jsonl'
    ]
    data_sets = []
    for s in sources:
        data_sets += from_jsonl(s)

    gating_model = train_with_entropy(
        data=data_sets,
        embed_dim=768,
        num_epochs=1,
        lambda_penalty=1.4,
        lambda_entropy=1.5,
        learning_rate=0.001
    )
    test_samples = from_jsonl("data_sets/test_unlabeled/mushroom.en-tst.v1.jsonl")
    zipped = get_training_results(gating_model, test_samples)
    zipped_spread = spread_all(zipped)
    spans = generate_spans(zipped_spread)
    print(spans)