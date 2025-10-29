from torch import nn
from HGNN import DualHGNN
from LLM import LLM
from MoE import MoE


class Model(nn.Module):
    def __init__(self, time_dim, news_dim, timestamps_dim, llm_hidden_size, N, T, dim, E1, E2,
                 llm_ckp_dir, device,
                 num_Local_HGConv=1,
                 num_heads_MHSA=2, num_Global_HGConv=1,
                 market_dim=32,
                 num_market_experts=9, top_k_market=3,
                 num_industry_experts=3, top_k_industry=1,
                 style_dim=32,
                 alpha=1e-4, beta=1e-4, noisy=False,
                 dropout=0.1, eps=1e-6):
        super().__init__()

        ### Feature embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            ) # 1. time embedding FFN

        self.news_embedding = nn.Sequential(
            nn.Linear(news_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            ) # 2. news embedding FFN

        self.timestamp_embedding = nn.Sequential(
            nn.Linear(timestamps_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        ) # 3. timestamp embedding FFN


        self.DualHGNN = DualHGNN(N=N, T=T, dim=dim, E1=E1, E2=E2, num_Local_HGConv=num_Local_HGConv,
                 num_heads_MHSA=num_heads_MHSA, num_Global_HGConv=num_Global_HGConv, dropout=dropout, eps=eps)


        self.LLM = LLM(dim=dim, llm_ckp_dir=llm_ckp_dir, device=device, dropout=dropout)

        self.MoE = MoE(N=N, T=T, llm_hidden_size=llm_hidden_size, dim=dim, E2=E2, market_dim=market_dim,
                 num_market_experts=num_market_experts, top_k_market=top_k_market,
                 num_industry_expert=num_industry_experts, top_k_industry=top_k_industry,
                 style_dim=style_dim,
                 alpha=alpha, beta=beta, dropout=dropout, noisy=noisy)


    def forward(self, x_time, x_news, timestamps):
        ### Feature embedding
        # 1. numerical feature embedding
        x_time = self.time_embedding(x_time) # [N, T, time_dim]->[N, T, dim]
        # 2. news embedding
        x_news = self.news_embedding(x_news)  # [N, T, news_dim]->[N, T, dim]
        # 3. timestamp embedding
        timestamps = self.timestamp_embedding(timestamps)
        # 4. positional encoding for numerical feature using timestamp embedding
        x_time = x_time + timestamps

        ## DualHGNN: local hypergraph and global hypergraph conv
        x_time, x_news, H_glob = self.DualHGNN(x_time, x_news)

        ## LLM processing
        x = self.LLM(x_time, x_news)

        ## MoE
        x, aux_loss_market, aux_loss_industry = self.MoE(x, H_glob)

        return x, aux_loss_market, aux_loss_industry