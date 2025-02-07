#!/usr/bin/env python3

import os
import re
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import subprocess
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")


# -----------------------------
# 1. EMBEDDING FUNCTIONS
# -----------------------------
def embed_input_text(text):
    return torch.from_numpy(sentence_transformer_model.encode(text))


def embed_output_token(token):
    return torch.from_numpy(sentence_transformer_model.encode(token))

def cached_data_sets():
    return "cached_data_sets"

 #'data_sets/test_labeled/mushroom.zh-tst.v1.jsonl'
 #'data_sets/validation/mushroom.ar-val.v2.jsonl'

def from_jsonl(jsonl_path):
    dataset = []
    idx = 1
    exceptions = []
    mapping_mismatch = 0
    path_split = jsonl_path.split("/")
    file_name = path_split[len(path_split)-1]
    parent_dir = path_split[0]
    cache = False
    if len(path_split)>1:
        cache_path = jsonl_path.replace(parent_dir, cached_data_sets())
    else:
        cache_path = cached_data_sets()+"/"+jsonl_path

    cache_path = cache_path.replace(file_name, "")
    cache_file = cache_path+"/"+file_name
    if os.path.exists(cache_file):
        jsonl_path = cache_file
        cache = True
    else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

    with open(jsonl_path, 'r', encoding='utf-8') as f, open(cache_file, 'a+', encoding='utf-8') as cache_f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            # Extract needed fields
            lang = data.get("lang", "").lower()
            id = data.get("id", f"train-{lang}-{idx}")
            model_input = data.get("model_input", "")
            model_id = data.get("model_id", "")
            output_tokens = data.get("model_output_tokens", [])
            output_logits = data.get("model_output_logits", [])
            output_text = data.get("model_output_text", "")
            # If cache of dataset is not available, perform data proccessing and save into cache
            if not cache:
                # Do some functional style black magic to fix CA data points (CA dataset was messy)
                if lang=="ca" and id.split("-")[:2]==["tst", "ca"]:
                    output_tokens = "".join(list(filter(lambda t: t!='[' and t!=']' and t!="'", output_tokens))).split(", ")
                    output_logits = list(map(float, "".join(list(filter(lambda l: l!='[' and l!=']' and l!="'", output_logits))).split(", ")))
                
                # Skip if invalid
                if not model_id or not output_tokens or not output_logits or not output_text:
                    continue
                
                # Filter technical tokens and their corresponding logits
                regex = re.compile(r'\<.*\>$')
                filtered = [(t, l) for (t, l) in zip(output_tokens, output_logits) if not regex.match(t)]
                output_tokens = [t for (t, _) in filtered]
                output_logits = [l for (_, l) in filtered]
                
                if model_id=="TheBloke/Mistral-7B-Instruct-v0.2-GGUF":
                    model_id="mistralai/Mistral-7B-Instruct-v0.2"

                if model_id=="AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-gguf":
                    model_id="AI-Sweden-Models/gpt-sw3-6.7b-v2"

                # Generate offset mapping for the specific model in the data point
                offset_mapping = []
                if id.split("-")[0]!="train":
                    try:
                        if model_id=='internlm/internlm2-chat-7b':
                            raise RuntimeError(f"use_fast=False")
                        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
                        tokenized_output = tokenizer(output_text, return_offsets_mapping=True)
                        offset_mapping = tokenized_output['offset_mapping']
                    except Exception as e:
                        exceptions.append((id, model_id, e))
                        #print(f"{id} {model_id} {e}")

                    # Clean (0, 0) technical mappings and other (x, y) zero character mappings, where x==y
                    offset_mapping = list(filter(lambda o: o[0]!=o[1], offset_mapping))

                    if not offset_mapping and lang=="ar":
                        output_tokens = [t[:max(1, len(t)//2)] for t in output_tokens]

                cached_json = {
                    "lang": lang,
                    "id": id,
                    "model_input": model_input,
                    "model_id": model_id,
                    "model_output_tokens": output_tokens,
                    "model_output_logits": output_logits,
                    "model_output_text": output_text,
                    "model_offset_mapping": offset_mapping
                }
                json_line = json.dumps(cached_json)
                cache_f.write(json_line + '\n')
            else:
                offset_mapping = data.get("model_offset_mapping", [])

                if len(offset_mapping)!=len(output_tokens):
                    mapping_mismatch += 1
                    #print(f"{id} mapping mismatch:")
                    #print(offset_mapping)
                    #print(output_tokens)
            dataset.append((id, model_input, output_text, offset_mapping, output_tokens, output_logits))
            idx += 1
    if dataset[0][0].split("-")[0]!="train":
        print(f"{jsonl_path} mapping mismatch count: {mapping_mismatch} ({len(exceptions)} exceptions)")

    return dataset


def labels_from_jsonl(jsonl_path):
    labels = []
    idx = 1

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data:
                continue
            # Extract needed fields
            lang = data.get("lang", "").lower()
            id = data.get("id", f"train-{lang}-{idx}")
            soft_labels = data.get("soft_labels", "?")
            hard_labels = data.get("hard_labels", "?")
            if soft_labels == "?" or hard_labels == "?":
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
    for (id, model_input, output_text, offset_mapping, output_tokens, output_logits) in dataset:
        log_probs = torch.tensor(output_logits, dtype=torch.float)
        gating_probs = gating_net(log_probs).detach().cpu().numpy()
        result.append((id, output_text, offset_mapping, output_tokens, gating_probs))
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

        for (_, model_input, output_text, offset_mapping, output_tokens, output_logits) in dataset:
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
    for (id, output_text, offset_mapping, output_tokens, gating_probs) in zipped:
        new_zipped.append((id, output_text, offset_mapping, output_tokens, spread_gating_probs(gating_probs, threshold, spread_factor)))

    #returns a list which contains first the original zipped and then with spread probs
    return new_zipped
    
def generate_span_groups(zipped, threshold):
    span_groups = []
    for (id, output_text, offset_mapping, output_tokens, gating_probs) in zipped:
        i = 0
        t = 0
        text_len = len(output_text)
        span = None
        span_group = {'id': id, 'spans': []}
        for token, gating_prob in zip(output_tokens, gating_probs):
            if offset_mapping and t>=len(offset_mapping):
                continue
            token_len = len(token) if not offset_mapping else abs(offset_mapping[t][0]-offset_mapping[t][1])
            if gating_prob >= threshold:
                if span is None:
                    span = {
                        'start': min(i, text_len),
                        'end': min(i + token_len, text_len),
                        'prob': gating_prob,
                        'tokens': [token]
                    }
                else:
                    span['end'] = min(i, text_len)
                    span['prob'] += gating_prob
                    span['tokens'].append(token)
                t += 1
            else:
                if span is not None:
                    span['prob'] /= (t + 1)
                    span_group['spans'].append(span)
                    span = None
                t = 0
            i += token_len
        span_groups.append({"id": id, "spans": list(filter(lambda s: s['start']!=s['end'], span_group['spans']))})

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

def get_unique_filename(filename, t_cfg, prob_thresh):
    base, ext = os.path.splitext(filename)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}-ne_{t_cfg['num_epochs']}_lp_{t_cfg['lambda_penalty']}_le_{t_cfg['lambda_entropy']}_lr_{t_cfg['learning_rate']}_pt_{prob_thresh}-{current_time}{ext}"
    return new_filename

def export_to_jsonl(predictions, jsonl_path):
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            json_line = json.dumps(prediction)
            f.write(json_line + '\n')

def format_checker(ref_dir, pred_file):
    subprocess.run([sys.executable, "participant_kit/format_checker.py", ref_dir, pred_file])

def scorer(ref_file, pred_file, output_file):
    subprocess.run([sys.executable, "participant_kit/scorer.py", ref_file, pred_file, output_file])
                
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

        error = 1.0 - correct / len(gold_labels)
    return error



def main(args):

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    run_dir = f"run-@{current_time}"
    output_path += "/" + run_dir
    os.makedirs(output_path)

    score_path = 'scores/o1_toy/'+run_dir
    if not os.path.exists(score_path):
        os.makedirs(score_path)

    score_path_classic = score_path + "/classic"
    if not os.path.exists(score_path_classic):
        os.makedirs(score_path_classic)

    score_path_spread = score_path + "/spread"
    if not os.path.exists(score_path_spread):
        os.makedirs(score_path_spread)

    output_path_classic = output_path + "/classic"
    if not os.path.exists(output_path_classic):
        os.makedirs(output_path_classic)

    output_path_spread = output_path + "/spread"
    if not os.path.exists(output_path_spread):
        os.makedirs(output_path_spread)

    errors_classic = {}
    errors_spread = {}

    last_infix = ""

    for path, lang in zip(args.test_path, args.test_lang):

        infix = path.split('-')[1][:3]

        test_samples = from_jsonl(path)
        zipped = get_training_results(gating_model, test_samples)
        zipped_spread = spread_all(zipped, spread_threshold, spread_factor)
        span_groups = generate_span_groups(zipped_spread, prob_threshold)
        predictions_classic, predictions_spread = generate_predictions(span_groups)

        gold_labels = labels_from_jsonl(path)

        error_classic, error_spread = calculate_error_pair(predictions_classic, predictions_spread, gold_labels)
        errors_classic[infix+"-"+lang] = error_classic
        errors_spread[infix+"-"+lang] = error_spread

        infix_path_classic = output_path_classic+f"/{infix}/"
        infix_path_spread = output_path_spread+f"/{infix}/"

        infix_score_path_classic = score_path_classic+f"/{infix}/"
        infix_score_path_spread = score_path_spread+f"/{infix}/"

        if infix != last_infix:
            if not os.path.exists(infix_path_classic):
                os.makedirs(infix_path_classic)
            if not os.path.exists(infix_path_spread):
                os.makedirs(infix_path_spread)
            if not os.path.exists(infix_score_path_classic):
                os.makedirs(infix_score_path_classic)
            if not os.path.exists(infix_score_path_spread):
                os.makedirs(infix_score_path_spread)
        last_infix = infix

        unique_filename_classic = get_unique_filename(f"{lang}-pred-{infix}.jsonl", train_config, prob_threshold)
        unique_filename_spread = get_unique_filename(f"{lang}-pred-{infix}_spread.jsonl", train_config, prob_threshold)

        infix_path_classic += unique_filename_classic
        infix_path_spread += unique_filename_spread

        export_to_jsonl(predictions_classic, infix_path_classic)
        export_to_jsonl(predictions_spread, infix_path_spread)

        ref_dir = path.split("mushroom")[0]

        print(f"Checking format: {unique_filename_classic}...{format_checker(ref_dir, infix_path_classic)}")
        print(f"Checking format: {unique_filename_spread}...{format_checker(ref_dir, infix_path_spread)}")

        score_classic = infix_score_path_classic+unique_filename_classic+"_score.txt"
        score_spread = infix_score_path_spread+unique_filename_spread+"_score.txt"

        print(f"Scoring: {unique_filename_classic}...{scorer(path, infix_path_classic, score_classic)}")
        print(f"Scoring: {unique_filename_spread}...{scorer(path, infix_path_spread, score_spread)}")

    print(f"Errors classic: {errors_classic}")
    print(f"Errors spread:  {errors_spread}")
    return errors_classic, errors_spread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', nargs='+', type=str,
        default=[
            'data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl'#,
            #'data_sets/train_unlabeled/mushroom.es-train_nolabel.v1.jsonl',
            #'data_sets/train_unlabeled/mushroom.fr-train_nolabel.v1.jsonl',
            #'data_sets/train_unlabeled/mushroom.zh-train_nolabel.v1.jsonl',
            #'data_sets/test_labeled/mushroom.ar-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.ca-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.cs-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.de-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.en-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.es-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.eu-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.fa-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.fi-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.fr-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.hi-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.it-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.sv-tst.v1.jsonl',
            #'data_sets/test_labeled/mushroom.zh-tst.v1.jsonl',
            #'data_sets/validation/mushroom.ar-val.v2.jsonl',
            #'data_sets/validation/mushroom.de-val.v2.jsonl',
            #'data_sets/validation/mushroom.en-val.v2.jsonl',
            #'data_sets/validation/mushroom.es-val.v2.jsonl',
            #'data_sets/validation/mushroom.fi-val.v2.jsonl',
            #'data_sets/validation/mushroom.fr-val.v2.jsonl',
            #'data_sets/validation/mushroom.hi-val.v2.jsonl',
            #'data_sets/validation/mushroom.it-val.v2.jsonl',
            #'data_sets/validation/mushroom.sv-val.v2.jsonl',
            #'data_sets/validation/mushroom.zh-val.v2.jsonl'
        ],
        help="Path to the training data")
    parser.add_argument('--test_path', nargs='+', type=str,
        default=[     
            'data_sets/test_labeled/mushroom.ar-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.ca-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.cs-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.de-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.en-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.es-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.eu-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.fa-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.fi-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.fr-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.hi-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.it-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.sv-tst.v1.jsonl',
            'data_sets/test_labeled/mushroom.zh-tst.v1.jsonl',
            'data_sets/validation/mushroom.ar-val.v2.jsonl',
            'data_sets/validation/mushroom.de-val.v2.jsonl',
            'data_sets/validation/mushroom.en-val.v2.jsonl',
            'data_sets/validation/mushroom.es-val.v2.jsonl',
            'data_sets/validation/mushroom.fi-val.v2.jsonl',
            'data_sets/validation/mushroom.fr-val.v2.jsonl',
            'data_sets/validation/mushroom.hi-val.v2.jsonl',
            'data_sets/validation/mushroom.it-val.v2.jsonl',
            'data_sets/validation/mushroom.sv-val.v2.jsonl',
            'data_sets/validation/mushroom.zh-val.v2.jsonl'
        ],
        help="Path to the testing data")
    parser.add_argument('--test_lang', nargs='+', type=str,
        default=['ar', 'ca', 'cs', 'de', 'en', 'es', 'eu', 'fa', 'fi', 'fr', 'hi', 'it', 'sv', 'zh', 'ar', 'de', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh'],
        help="List of test languages")
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