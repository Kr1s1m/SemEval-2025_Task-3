#!/usr/bin/env python3

import os
import json
import argparse
import sentence_transformers
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer


sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")


# -----------------------------
# 1. EMBEDDING FUNCTIONS
# -----------------------------
def embed_input_text(text):
    return torch.from_numpy(sentence_transformer_model.encode(text))


def embed_output_token(token):
    return torch.from_numpy(sentence_transformer_model.encode(token))


def from_jsonl(jsonl_path, gold=False):
    dataset = []
    idx = 1
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            # Extract needed fields
            lang = data.get("lang", "")
            id = data.get("id", f"train-{lang.lower()}-{idx}")
            model_input = data.get("model_input", "")
            if not gold:
                output_tokens = data.get("model_output_tokens", [])
                output_logits = data.get("model_output_logits", [])
                if not output_tokens or not output_logits:
                    continue
                dataset.append((id, model_input, output_tokens, output_logits))
            else :
                soft_labels = data.get("soft_labels", [])
                hard_labels = data.get("hard_labels", [])
                if not soft_labels and not hard_labels:
                    continue
                dataset.append((id, soft_labels, hard_labels))

            # Skip if invalid

            idx += 1

    return dataset


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


# ----------------------------------
# 3. MAIN TRAINING LOOP V2 (ENTROPY)
# ----------------------------------

def train_with_entropy(
    dataset,
    num_epochs=3,
    lambda_penalty=0.01,
    lambda_entropy=0.01,
    learning_rate=0.01,
    embed_dim=768
):
    """
    Training loop:
    - Replaces L2 with cross-entropy alignment for consistency.
    - Adds gating entropy penalty for pushing gating to 0 or 1.
    - Optionally, you can also keep the sum(g_i) penalty if you want.
    """
    gating_net = GatingNetwork()
    optimizer = optim.Adam(gating_net.parameters(), learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for (_, model_input, output_tokens, output_logits) in dataset:
            log_probs = torch.tensor(output_logits, dtype=torch.float)
            gating_probs = gating_net(log_probs)

            # Embed input text => distribution
            input_embed = embed_input_text(model_input)
            input_dist = embed_to_distribution(input_embed)

            # Soft-mask output
            masked_output_embed = torch.zeros(embed_dim)
            for i, token in enumerate(output_tokens):
                token_embed = embed_output_token(token)
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

    #new_probs[0] *= 0.01

    return new_probs

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def spread_all(zipped, threshold=0.8, spread_factor=0.1):
    new_zipped = zipped.copy()
    for (id, tokens, gating_probs) in zipped:
        new_zipped.append((id, tokens, spread_gating_probs(gating_probs, threshold, spread_factor)))

    #returns a list which contains first the original zipped and then with spread probs
    return new_zipped
    
def generate_span_groups(zipped, threshold):
    span_groups = []
    for (id, tokens, gating_probs) in zipped:
        i = 0
        span = None
        span_group = {'id': id, 'spans': []}
        for token, gating_prob in zip(tokens, gating_probs):
            if gating_prob >= threshold:
                if span is None:
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
                    span_group['spans'].append(span)
                    span = None
            i+=len(token)
        span_groups.append(span_group)

    return span_groups

def generate_predictions(span_groups):
    predictions = []
    for span_group in span_groups:
        prediction = {
            'id': span_group['id'],
            'soft_labels': [],
            'hard_labels': []
        }
        for span in span_group['spans']:
            if not span: 
                continue
            soft_label = {
                'start': span['start'],
                'end': span['end'],
                'prob': float(span['prob'])
            }
            prediction['soft_labels'].append(soft_label)
            if span['prob'] >= 0.5:
                hard_label = [soft_label['start'], soft_label['end']]
                prediction['hard_labels'].append(hard_label)
        predictions.append(prediction)
    
    #returns predictions_classic, predictions_spread tuple of lists - check spread_all and split_list
    return split_list(predictions)

def get_unique_filename(filename, t_cfg):
    base, ext = os.path.splitext(filename)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}-ne_{t_cfg['num_epochs']}_lp_{t_cfg['lambda_penalty']}_le_{t_cfg['lambda_entropy']}_lr_{t_cfg['learning_rate']}-{current_time}{ext}"
    return new_filename

def export_to_jsonl(predictions, jsonl_path):
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            json_line = json.dumps(prediction)
            f.write(json_line + '\n')
                
def calculate_error_pair(silver_labels_classic, silver_labels_spread, gold_labels):
    error_classic = calculate_error(silver_labels_classic, gold_labels)
    error_spread = calculate_error(silver_labels_spread, gold_labels)

    return error_classic, error_spread

def calculate_error(silver_labels, gold_labels):
    error = -1.0
    correct = 0.0

    if len(silver_labels) == len(gold_labels):
        for silver_label, (id, soft_labels, hard_labels) in zip(silver_labels, gold_labels):
            for silver_span, gold_span in zip(silver_label['soft_labels'], soft_labels):
                gold_len = abs(gold_span['start']-gold_span['end'])+1
                if silver_span['start'] == gold_span['start']:
                    if silver_span['end'] == gold_span['end']:
                        correct += 1.0
                    else:
                        correct += 1.0 - min(1.0, (abs(silver_span['end']-gold_span['end'])+1)/(gold_len))
                elif silver_span['start'] > gold_span['start']:
                    if silver_span['start'] > gold_span['end']:
                        correct += 0.0
                    elif silver_span['end'] > gold_span['end']:
                        correct += 1.0 - min(1.0, (abs(silver_span['start']-gold_span['end'])+1)/(gold_len))
                    else: #silver_span['end'] <= gold_span['end']
                        correct += 1.0 - min(1.0, (abs(silver_span['start']-silver_span['end'])+1)/(gold_len))
                else: #silver_span['start'] < gold_span['start']
                    if silver_span['end'] < gold_span['start']:
                        correct += 0.0
                    elif silver_span['end'] <= gold_span['end']:
                        correct += 1.0 - min(1.0, (abs(gold_span['start']-silver_span['end'])+1)/(gold_len))
                    else: #silver_span['end'] > gold_span['end']
                        correct += 1.0 - min(1.0, (abs(silver_span['start']-silver_span['end'])+1)/(gold_len))

        error = correct / len(gold_labels)
    return error



def main(args):
    # Example usage of soft gating
    #train_soft_gating(data="data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl", embed_dim=768, num_epochs=3)

    lang = args.test_lang

    sources = args.data_path
    train_samples = []
    for s in sources:
        train_samples += from_jsonl(s)

    train_config = {
        'num_epochs': args.num_epochs,
        'lambda_penalty': args.lambda_penalty,
        'lambda_entropy': args.lambda_entropy,
        'learning_rate': args.learning_rate
    }

    gating_model = train_with_entropy(
        train_samples,
        train_config['num_epochs'],
        train_config['lambda_penalty'],
        train_config['lambda_entropy'],
        train_config['learning_rate']
    )

    prob_threshold = args.prob_threshold

    spread_threshold = args.spread_threshold
    spread_factor = args.spread_factor

    test_samples = from_jsonl(args.test_path)
    zipped = get_training_results(gating_model, test_samples)
    zipped_spread = spread_all(zipped, spread_threshold, spread_factor)
    span_groups = generate_span_groups(zipped_spread, prob_threshold)
    predictions_classic, predictions_spread = generate_predictions(span_groups)
    #print(predictions_classic)
    #print(predictions_spread)

    gold_labels = from_jsonl(args.test_path, gold=True)

    error_classic, error_spread = calculate_error_pair(predictions_classic, predictions_spread, gold_labels)
    print(f"Error classic:  {error_classic} \n Error spread:  {error_spread}")

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    export_to_jsonl(predictions_classic, output_path+get_unique_filename(f"{lang}-pred.jsonl", train_config))
    export_to_jsonl(predictions_spread, output_path+get_unique_filename(f"{lang}-pred_spread.jsonl", train_config))

    return error_classic, error_spread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', nargs='+', type=str,
        default=[
            'data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl',
            'data_sets/test_unlabeled/mushroom.en-tst.v1.jsonl'
        ],
        help="Path to the training data")
    parser.add_argument('--test_path', type=str, default='data_sets/validation/mushroom.en-val.v2.jsonl', help="Path to the testing data")
    parser.add_argument('--test_lang', type=str, default='en')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lambda_penalty', type=float, default=1.2)
    parser.add_argument('--lambda_entropy', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--prob_threshold', type=float, default=0.6)
    parser.add_argument('--spread_threshold', type=float, default=0.8)
    parser.add_argument('--spread_factor', type=float, default=0.1)
    parser.add_argument('--output_path', type=str, default='test/results/o1_toy/', help="Path to save the predictions at")
    args = parser.parse_args()
    main(args)