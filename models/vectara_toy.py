import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

###############################################################################
# 1) Dummy Embedding + GatingNetwork with Vectara Probability
###############################################################################

def embed_input_text(text, embed_dim=8):
    # Dummy random embedding for demonstration
    rnd_state = np.random.RandomState(abs(hash(text)) % (2**31))
    return torch.tensor(rnd_state.randn(embed_dim), dtype=torch.float)

def embed_output_token(token, embed_dim=8):
    rnd_state = np.random.RandomState(abs(hash(token)) % (2**31))
    return torch.tensor(rnd_state.randn(embed_dim), dtype=torch.float)

def vectara_score(span, input_text):
    """
    Pseudo-function that queries Vectara with (span, input_text) and returns
    a float in [0,1] indicating how relevant/factually correct that span is.
    In reality, you'd make an API call or use a client library.
    """
    # For demonstration, return a random probability.
    # Replace with an actual Vectara API request (see docs).
    return np.random.rand()

class GatingNetworkWithVectara(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        # We'll define a small MLP that takes [log_prob, vectara_prob, ... embeddings ...]
        # and outputs gating_prob in (0,1).
        
        input_size = 2 + embed_dim  # log_prob + vectara_prob + embed_dim
        hidden_size = 16
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # final = single logit
        )
    
    def forward(self, log_probs, vectara_probs, token_embeds):
        """
        Inputs:
          log_probs: (num_tokens,) float
          vectara_probs: (num_tokens,) float
          token_embeds: (num_tokens, embed_dim)
        
        Output:
          gating_probs: (num_tokens,) in [0,1]
        """
        # Combine the features
        # shape => (num_tokens, input_size)
        combined = torch.cat([
            log_probs.unsqueeze(1), 
            vectara_probs.unsqueeze(1),
            token_embeds
        ], dim=1)
        
        logits = self.mlp(combined)  # shape (num_tokens, 1)
        gating_probs = torch.sigmoid(logits).squeeze(1)  # (num_tokens,)
        return gating_probs

###############################################################################
# 2) Main Training Loop with a cross-entropy alignment + gating + Vectara
###############################################################################
def train_with_vectara(
    jsonl_path="data.jsonl",
    embed_dim=8,
    num_epochs=3,
    lambda_penalty=0.01
):
    """
    We'll:
      - Parse each example (input_text, output_tokens, output_logits)
      - Call vectara_score for each token's "span" (just the single token or small chunk)
      - Then do the gating logic + cross-entropy alignment
      - Add a gating penalty
    """
    gating_net = GatingNetworkWithVectara(embed_dim=embed_dim)
    optimizer = optim.Adam(gating_net.parameters(), lr=0.01)

    # 1. Load data
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
            # Convert to torch
            log_probs = torch.tensor(output_logits, dtype=torch.float)
            
            # 2) Query vectara_score for each token (in real code, you might batch this)
            vectara_scores = []
            for tok in output_tokens:
                # For demonstration, let's treat "span" as just the single token
                # or you might use a multi-token chunk around 'tok'
                vs = vectara_score(span=tok, input_text=model_input)
                vectara_scores.append(vs)
            vectara_probs = torch.tensor(vectara_scores, dtype=torch.float)
            
            # 3) Build token embeddings
            token_embs = []
            for tok in output_tokens:
                e = embed_output_token(tok, embed_dim=embed_dim)
                token_embs.append(e)
            token_embs = torch.stack(token_embs, dim=0)  # shape (num_tokens, embed_dim)
            
            # 4) Forward pass -> gating_probs
            gating_probs = gating_net(log_probs, vectara_probs, token_embs)
            # gating_probs: shape (num_tokens,)
            
            # 5) Build masked_output_embed
            #    (1 - gating_probs[i]) => how much we keep each token
            masked_output_embed = torch.zeros(embed_dim)
            for i, tok_embed in enumerate(token_embs):
                keep_weight = (1 - gating_probs[i])
                masked_output_embed += keep_weight * tok_embed

            # 6) Compute input_embed and do cross-entropy alignment
            input_embed = embed_input_text(model_input, embed_dim=embed_dim)
            # Convert to "distributions" if you're doing cross-entropy approach
            # or do a simpler L2 distance. We'll do L2 for brevity:
            loss_consistency = torch.mean((masked_output_embed - input_embed)**2)

            # 7) Gating penalty => we don't want gating_probs to be all 1
            loss_penalty = gating_probs.mean()
            
            loss = loss_consistency + lambda_penalty * loss_penalty
            
            # 8) Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  Avg loss this epoch: {total_loss/len(dataset):.4f}")

    print("\nFinal gating distribution for an example:")
    gating_net.eval()
    # Let's just show the gating for the first example
    if dataset:
        (model_input, output_tokens, output_logits) = dataset[0]
        log_probs = torch.tensor(output_logits, dtype=torch.float)
        vectara_scores = [vectara_score(tok, model_input) for tok in output_tokens]
        vectara_probs = torch.tensor(vectara_scores, dtype=torch.float)
        token_embs = torch.stack([embed_output_token(tok, embed_dim=embed_dim)
                                  for tok in output_tokens], dim=0)
        
        gating_probs = gating_net(log_probs, vectara_probs, token_embs).detach().numpy()
        for tok, gp in zip(output_tokens, gating_probs):
            print(f"  {tok} => gating_prob={gp:.3f}")

if __name__ == "__main__":
    train_with_vectara("data.jsonl", embed_dim=8, num_epochs=3, lambda_penalty=0.01)