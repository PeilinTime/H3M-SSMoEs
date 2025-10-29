from torch import nn
import torch
import torch.nn.functional as nnF


def sequence_wise_auxiliary_loss(batch_size, seq_len, num_experts, top_k, gating_output, indices, alpha):
    """
    Compute sequence-wise auxiliary loss for load balancing

    Args:
        gating_output: Router output probabilities [batch_size, seq_len, num_experts]
        indices: Selected expert indices [batch_size, seq_len, top_k]
        alpha: Balance factor

    Returns:
        auxiliary_loss: Scalar loss value
    """

    T = batch_size * seq_len  # Total tokens in sequence

    # Flatten batch and sequence dimensions
    flat_gating = gating_output.view(-1, num_experts)  # [T, Nr]
    flat_indices = indices.view(-1, top_k)  # [T, Kr]

    # Compute fi: fraction of tokens assigned to each expert
    fi = torch.zeros(num_experts, device=gating_output.device)
    for i in range(num_experts):
        # Count how many times expert i is selected
        expert_mask = (flat_indices == i).any(dim=-1).float()
        fi[i] = expert_mask.sum() * num_experts / (top_k * T)

    Pi = flat_gating.mean(dim=0)  # Average over all tokens

    # Compute balance loss
    auxiliary_loss = alpha * (fi * Pi).sum()

    return auxiliary_loss

# Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, dim, style_dim, dropout=0.1):
        super().__init__()

        # FFN
        self.net = nn.Sequential(
            nn.Linear(dim+style_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        # New: different experts have different style, and automatically learns different styles through training.
        self.style = nn.Parameter(torch.randn(1, style_dim)) # [1, style_dim]


    def forward(self, x): # x: [n, dim]
        # Concatenate stock feature x with expert style self.style
        x = torch.cat([x, self.style.expand(x.shape[0], -1)], dim=-1) # [n, dim+style_dim]
        return self.net(x) # [n, dim]


class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k, noisy=False):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.noisy = noisy
        if self.noisy:
            self.noise_linear =nn.Linear(dim, num_experts)

    def forward(self, x):
        logits = self.topkroute_linear(x)

        if self.noisy:
            noise_logits = self.noise_linear(x)
            #Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(logits)*nnF.softplus(noise_logits)
            logits = logits + noise

        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = nnF.softmax(sparse_logits, dim=-1)
        return router_output, indices

# create the sparse mixture of experts module for global market, or to evaluate the current market status
class SparseMarketMoE(nn.Module):
    def __init__(self, N, dim, market_dim=32, num_market_experts=9, top_k_market=3, style_dim=32,
                 dropout=0.1, alpha=1e-4, noisy=False):
        super(SparseMarketMoE, self).__init__()
        self.router = NoisyTopkRouter(dim+market_dim, num_experts=num_market_experts, top_k=top_k_market, noisy=noisy)
        self.experts = nn.ModuleList([Expert(dim=dim, style_dim=style_dim, dropout=dropout) for _ in range(num_market_experts)])
        self.top_k = top_k_market
        self.num_experts = num_market_experts

        self.Wr = nn.Sequential(nn.Linear(dim, market_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                )
        self.Wl = nn.Linear(N, 1)


        self.batch_size = N
        self.seq_len = 1
        self.alpha = alpha

    def forward(self, x): # x: [N, 1, dim]
        # Extract/Learn the current market information/status
        market = self.Wr(x) # [N, 1, market_dim]
        market = self.Wl(market.transpose(0, 2)).transpose(0, 2) # [N, 1, market_dim] -> [market_dim, 1, N] @ [N, 1] -> [market_dim, 1, 1] -> [1, 1, market_dim]

        market_expanded = market.expand(self.batch_size, -1, -1)  # Expands m to [N, 1, market_dim] without copying data
        x_market = torch.cat([x, market_expanded], dim=-1)  # [N, 1, dim+market_dim]. Connect the current market information/status with feature x.

        gating_output, indices = self.router(x_market)

        # Compute auxiliary loss for market: aux_loss_market
        aux_loss = sequence_wise_auxiliary_loss(self.batch_size, self.seq_len, self.num_experts, self.top_k, gating_output, indices, self.alpha)

        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)

            if selected_indices.numel() > 0:
                expert_input = flat_x[selected_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[selected_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                updates.index_add_(0, selected_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(self.batch_size, self.seq_len, -1) # x: [N, 1, dim]


        return final_output, aux_loss # x: [N, 1, dim], aux_loss



# create the sparse mixture of experts module for different industries
class SparseIndustryMoE(nn.Module):
    def __init__(self, N, dim, E2, num_industry_expert=3, top_k_industry=1,
                 style_dim=32, dropout=0.1, beta=1e-4, noisy=False):
        super(SparseIndustryMoE, self).__init__()
        self.router = NoisyTopkRouter(dim+E2, num_experts=num_industry_expert, top_k=top_k_industry, noisy=noisy)
        self.experts = nn.ModuleList([Expert(dim=dim, style_dim=style_dim, dropout=dropout) for _ in range(num_industry_expert)])
        self.top_k = top_k_industry
        self.num_experts = num_industry_expert

        self.industry = nn.Linear(E2, E2)

        self.batch_size = N
        self.seq_len = 1

        self.beta = beta


    def forward(self, x, H_glo): # x: [N, 1, dim]. H_glo:[N, E], from self.GlobalHGNN (GlobalHypergraph) in HGNN.py
        industry = self.industry(H_glo).unsqueeze(1) # [N, E] -> [N, 1, E]. industries transformation/projection.
        x_industry = torch.cat([x, industry], dim=-1)  # [N, 1, dim+E]. concatenate different industry statuses for each stock and feature x.

        gating_output, indices = self.router(x_industry)

        # Compute auxiliary loss for industry: aux_loss_industry
        aux_loss = sequence_wise_auxiliary_loss(self.batch_size, self.seq_len, self.num_experts, self.top_k, gating_output, indices, self.beta)

        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)

            if selected_indices.numel() > 0:
                expert_input = flat_x[selected_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[selected_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                updates.index_add_(0, selected_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(self.batch_size, self.seq_len, -1) # x: [N, 1, F]

        return final_output, aux_loss # x: [N, 1, dim], aux_loss...


class MoE(nn.Module):
    def __init__(self, N, T, llm_hidden_size, dim, E2, market_dim=32,
                 num_market_experts=9, top_k_market=3,
                 num_industry_expert=3, top_k_industry=1,
                 style_dim=32,
                 alpha=1e-4, beta=1e-4, dropout=0.1, noisy=False):
        super(MoE, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(T*llm_hidden_size, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.SparseMarketMoE = SparseMarketMoE(N=N, dim=dim, market_dim=market_dim, num_market_experts=num_market_experts,
                                               top_k_market=top_k_market, style_dim=style_dim,
                 dropout=dropout, alpha=alpha, noisy=noisy) # market MoE

        self.SparseIndustryMoE = SparseIndustryMoE(N=N, dim=dim, E2=E2, num_industry_expert=num_industry_expert, top_k_industry=top_k_industry,
                 style_dim=style_dim, dropout=dropout, beta=beta, noisy=noisy) # industry MoE

        self.market_embedding = nn.Linear(dim, dim)
        self.industry_embedding = nn.Linear(dim, dim)

        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2), # output logits
            )

    def forward(self, x, H_glob): # x: [N, T, llm_hidden_size], H_glob: [N, E]
        x = x.flatten(start_dim=1).unsqueeze(1) # [N, T*llm_hidden_size] -> [N, 1, T*llm_hidden_size]
        x = self.embedding(x) # [N, 1, dim]

        residual = x
        x = self.norm(x)

        '''
        Two type of MoE: one for the whole market (concatenate shared market information for all stocks), 
        one for different industry (concatenate different industry for each stock)
        '''
        x_market, aux_loss_market = self.SparseMarketMoE(x) # x_market: [N, 1, dim], ...
        x_industry, aux_loss_industry = self.SparseIndustryMoE(x, H_glob) # x_industry: [N, 1, dim], ...

        out = nnF.relu(self.market_embedding(x_market) + self.industry_embedding(x_industry)) # [N, 1, dim] # fuse the two type of MoE results
        out = self.dropout(out)
        x = residual + out

        x = self.output(x).squeeze(1) # [N, 2]

        return x, aux_loss_market, aux_loss_industry