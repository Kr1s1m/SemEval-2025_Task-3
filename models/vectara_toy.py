import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

###############################################################################
# 1) Dummy Embedding, Gating with Vectara, and Some Utility Functions
###############################################################################

def embed_input_text(text, embed_dim=8):
    """
    Replace with your real embedding code (e.g., sentence_transformer_model.encode).
    Here we just return a random vector for demonstration.
    """
    rnd_state = np.random.RandomState(abs(hash(text)) % (2**31))
    return torch.tensor(rnd_state.randn(embed_dim), dtype=torch.float)

def embed_output_token(token, embed_dim=8):
    """
    Also replace with real embedding logic.
    """
    rnd_state = np.random.RandomState(abs(hash(token)) % (2**31))
    return torch.tensor(rnd_state.randn(embed_dim), dtype=torch.float)

def vectara_score(span, input_text):
    """
    Pseudo-function returning a random probability in [0,1].
    In practice, you'd call the Vectara API with (span, input_text) 
    and parse the relevance/correctness probability it returns.
    """
    return np.random.rand()

def embed_to_distribution(embedding):
    """
    Convert an embedding vector into a probability distribution via softmax.
    """
    eps = 1e-8
    exp_emb = torch.exp(embedding)
    sum_exp = torch.sum(exp_emb) + eps
    return exp_emb / sum_exp

def binary_entropy(g):
    """
    Encourages gating probabilities to be near 0 or 1 by penalizing mid-values.
    If we add 'gating_entropy' to the loss, we are *minimizing* gating entropy,
    making gating more 'discrete.'
    """
    eps = 1e-8
    return - (g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))


###############################################################################
# 2) Gating Network That Incorporates Vectara Probability
###############################################################################
class GatingNetworkWithVectara(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        # We'll define a small MLP that takes 
        # [log_prob, vectara_prob, token_embedding (embed_dim)] -> gating_prob
        input_size = 2 + embed_dim  # log_prob + vectara_prob + embed_dim
        hidden_size = 16
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # final -> single logit
        )
    
    def forward(self, log_probs, vectara_probs, token_embs):
        """
        Inputs:
          log_probs: shape (num_tokens,)
          vectara_probs: shape (num_tokens,)
          token_embs: shape (num_tokens, embed_dim)
        Returns:
          gating_probs: shape (num_tokens,) in [0,1]
        """
        # Combine features for each token:
        # shape -> (num_tokens, input_size)
        combined = torch.cat([
            log_probs.unsqueeze(1),
            vectara_probs.unsqueeze(1),
            token_embs
        ], dim=1)
        
        logits = self.mlp(combined)        # (num_tokens, 1)
        gating_probs = torch.sigmoid(logits).squeeze(1)  # (num_tokens,)
        return gating_probs

###############################################################################
# 3) Main Training Loop: Cross-Entropy Alignment + Gating Entropy + Vectara
###############################################################################
def train_with_vectara_and_entropy(
    dataset,
    embed_dim=8,
    num_epochs=3,
    lambda_penalty=0.01,
    lambda_entropy=0.01,
    learning_rate=0.01
):
    """
    - We incorporate Vectara probabilities (relevance/correctness) as an input feature.
    - We do cross-entropy alignment between masked output embedding and the input embedding.
    - We add gating entropy to encourage gating to be near 0 or 1.
    - We also keep a simple penalty on gating mean to avoid gating out everything.
    
    Args:
      dataset: a list of (model_input, output_tokens, output_logits) examples
      embed_dim: dimension of your embeddings
      num_epochs, lambda_penalty, lambda_entropy, learning_rate: hyperparameters
    """
    gating_net = GatingNetworkWithVectara(embed_dim=embed_dim)
    optimizer = optim.Adam(gating_net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for (model_input, output_tokens, output_logits) in dataset:
            # 1) Convert log_probs to a tensor
            log_probs = torch.tensor(output_logits, dtype=torch.float)
            
            # 2) Obtain vectara_probs for each token
            vectara_vals = []
            for tok in output_tokens:
                # For demonstration, we consider 'tok' as the 'span'.
                # If you want multi-token spans, adjust accordingly.
                v_prob = vectara_score(tok, model_input)
                vectara_vals.append(v_prob)
            vectara_probs = torch.tensor(vectara_vals, dtype=torch.float)
            
            # 3) Build the token embeddings
            token_embs_list = []
            for tok in output_tokens:
                token_emb = embed_output_token(tok, embed_dim=embed_dim)
                token_embs_list.append(token_emb)
            token_embs = torch.stack(token_embs_list, dim=0)  # (num_tokens, embed_dim)

            # 4) Forward pass -> gating_probs
            gating_probs = gating_net(log_probs, vectara_probs, token_embs)

            # 5) Build masked_output_embed
            #    If gating_prob is near 1, we remove that token (like a "mask-out" approach).
            masked_output_embed = torch.zeros(embed_dim)
            for i, tok_emb in enumerate(token_embs):
                keep_weight = (1 - gating_probs[i])
                masked_output_embed += keep_weight * tok_emb

            # 6) Convert input_embed & masked_output_embed to distributions & do cross-entropy
            input_embed = embed_input_text(model_input, embed_dim=embed_dim)
            input_dist = embed_to_distribution(input_embed)
            masked_dist = embed_to_distribution(masked_output_embed)

            loss_consistency = -torch.sum(input_dist * torch.log(masked_dist + 1e-8))

            # 7) Gating penalty => average gating to discourage gating all tokens out
            loss_penalty = gating_probs.mean()

            # 8) Gating entropy => encourage gating to be near 0 or 1
            gating_ent = binary_entropy(gating_probs)
            gating_ent_mean = gating_ent.mean()

            # Final loss
            loss = (
                loss_consistency
                + lambda_penalty * loss_penalty
                + lambda_entropy * gating_ent_mean
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"  Avg loss: {avg_loss:.4f}")

    print("\nFinished training with cross-entropy + gating entropy + Vectara.")
    return gating_net


###############################################################################
# 4) Example Usage
###############################################################################
def example_usage():
    # Suppose you have some unlabeled data in the format:
    # dataset = [
    #   (model_input, [output_tokens], [output_logits]),
    #   ...
    # ]
    # We simulate some dummy data here:
    dataset = [
        ("Question about mushrooms", ["Yes", "No", "I", "think", "so"], [-1.2, 0.5, 0.1, -0.9, -2.3]),
        ("Another input text", ["Possible", "answer"], [0.9, -0.8])
    ]

    gating_net = train_with_vectara_and_entropy(
        dataset,
        embed_dim=8,
        num_epochs=3,
        lambda_penalty=0.01,
        lambda_entropy=0.01,
        learning_rate=0.01
    )

    # After training, we can test gating:
    gating_net.eval()
    test_input = "Mushroom question"
    test_tokens = ["Yes", "No", "Maybe"]
    test_logits = [-0.5, 1.1, -0.3]

    with torch.no_grad():
        log_probs = torch.tensor(test_logits, dtype=torch.float)
        vectara_vals = [vectara_score(tok, test_input) for tok in test_tokens]
        vectara_probs = torch.tensor(vectara_vals, dtype=torch.float)
        token_embs = torch.stack([embed_output_token(t) for t in test_tokens], dim=0)

        gating_probs = gating_net(log_probs, vectara_probs, token_embs).numpy()
        for tok, gp in zip(test_tokens, gating_probs):
            print(f"Token='{tok}', gating_prob={gp:.3f}")

# Uncomment to run example:
# example_usage()
