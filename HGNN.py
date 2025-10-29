import torch
from torch import nn
import torch.nn.functional as nnF


def jensen_shannon_divergence(matrix):
    """
    Compute Jensen-Shannon Divergence between each pair of column vectors.

    Args:
        matrix: Tensor of shape [N, E] where N is the dimension of each vector
                and E is the number of vectors (columns)
        epsilon: Small value to avoid log(0) and division by zero

    Returns:
        jsd_matrix: Tensor of shape [E, E] where element (i,j) is JSD(col_i, col_j)
    """
    # Ensure the matrix is a probability distribution along each column
    # (non-negative and sums to 1)

    # Efficient computation using broadcasting
    # Reshape for broadcasting: [N, E, 1] and [N, 1, E]
    P = matrix.unsqueeze(2)  # Shape: [N, E, 1]
    Q = matrix.unsqueeze(1)  # Shape: [N, 1, E]

    # Compute M = (P + Q) / 2 for all pairs
    M = (P + Q) / 2.0  # Shape: [N, E, E]

    # Compute KL divergences
    # KL(P||M) for all pairs
    kl_pm = (P * (torch.log(P) - torch.log(M))).sum(dim=0)  # Shape: [E, E]

    # KL(Q||M) for all pairs
    kl_qm = (Q * (torch.log(Q) - torch.log(M))).sum(dim=0)  # Shape: [E, E]

    # Jensen-Shannon Divergence
    jsd = (kl_pm + kl_qm) / 2.0

    return jsd


class HypergraphConvolution(nn.Module):
    """
    Hypergraph convolution operation.

    Performs: X' = σ(H @ W @ H^T @ X @ Theta)
    where H is the incidence matrix, W is hyperedge weights,
    X is node features, and Theta is a learnable projection.

    Args:
        F (int): Feature dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn([dim, dim]))

    def forward(
            self,
            H: torch.Tensor,
            x: torch.Tensor,
            W: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward hypergraph convolution.

        Args:
            H: Incidence matrix [nodes, hyperedges]
            x: Node features [nodes, features]
            W: diagonal hyperedge weights [hyperedges, hyperedges]

        Returns:
            Updated node features [nodes, features]
        """
        x = nnF.elu(H @ W @ H.t() @ x @ self.theta)
        return x


class LocalHypergraph(nn.Module):
    def __init__(self, N, T, dim, E1, num_Local_HGConv=1, dropout=0.1, eps=1e-6):
        super().__init__()
        ### Compress-layer FFN, condensing N × T instances into E₁ hidden semantics.

        ## intra modality
        # numerical feature embedding
        self.intra_time = nn.Sequential(
            nn.Linear(N * T, E1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E1, E1),
        )
        # news embedding
        self.intra_news = nn.Sequential(
            nn.Linear(N * T, E1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E1, E1),
        )

        ## inter modality
        # news embedding
        self.time2news = nn.Sequential(
            nn.Linear(N * T, E1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E1, E1),
        )
        self.news2time = nn.Sequential(
            nn.Linear(N * T, E1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E1, E1),
        )

        self.dropout = nn.Dropout(dropout)

        self.fusion = nn.Sequential(
            nn.Linear(4 * E1, E1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E1, E1),
        )

        # Hypergraph convolution layers
        self.LocalHGConv_time = nn.ModuleList([
            HypergraphConvolution(dim)
            for _ in range(num_Local_HGConv)
        ])

        self.LocalHGConv_news = nn.ModuleList([
            HypergraphConvolution(dim)
            for _ in range(num_Local_HGConv)
        ])

        self.N = N
        self.T = T
        self.dim = dim
        self.eps = eps

    def forward(self, x_time, x_news):
        x_time = x_time.flatten(0, 1)  # [N*T, dim]

        x_news = x_news.flatten(0, 1)  # [N*T, dim]

        ### dimensionality reduction for N dimension
        ## inter-modality
        # numerical feature to numerical feature
        # A compression-layer FFN that condenses or extracts N × T instances into E₁ hidden semantic representations,
        # thereby reducing computational complexity.
        H_time1 = self.intra_time(x_time.permute(1, 0))  # [N*T,dim]->[dim,N*T]->[dim,E1].

        # Compute cosine similarity via matrix multiplication to obtain intermediate sub-hypergraph H_time
        H_time = torch.mm(x_time, H_time1)  # Shape: [N*T, E1]

        # news to news
        H_news1 = self.intra_news(x_news.permute(1, 0))  # [N*T,dim]->[dim,N*T]->[dim,E1]
        # Compute cosine similarity via matrix multiplication to obtain intermediate sub-hypergraph H_news
        H_news = torch.mm(x_news, H_news1)  # Shape: [N*T, E1]

        ## inter-modality
        # numerical feature to news
        H_news2 = self.time2news(x_news.permute(1, 0))  # [N*T,dim]->[dim,N*T]->[dim,E1]
        # Compute cosine similarity via matrix multiplication to obtain intermediate sub-hypergraph H_t2n
        H_t2n = torch.mm(x_time, H_news2)  # Shape: [N*T, E1]

        # news to numerical feature
        H_time2 = self.news2time(x_time.permute(1, 0))  # [N*T,dim]->[dim,N*T]->[dim,E1]
        # Compute cosine similarity via matrix multiplication to obtain intermediate sub-hypergraph H_n2t
        H_n2t = torch.mm(x_news, H_time2)  # Shape: [N*T, E1]

        # 4 sub-hypergraphs fusion to generate final local hypergraph
        H = self.fusion(torch.cat([H_time, H_news, H_t2n, H_n2t], dim=1))  # Shape: [N*T, 4*E1] -> [N*T, E1]
        H_mean = H.mean(dim=0, keepdim=True)
        H_std = H.std(dim=0, keepdim=True)
        H = (H - H_mean) / (H_std + self.eps)
        H = nnF.softmax(H, dim=0)

        # Jensen-Shannon Divergence weighting mechanism for each hyperedge (column in hypergraph)
        JSD = jensen_shannon_divergence(H)
        JSD_mean = JSD.mean(dim=0)
        # Z-score Normalization and create diagonal weight matrix
        normalize_JSD_mean = (JSD_mean - JSD_mean.mean()) / (JSD_mean.std() + self.eps)
        W = nnF.softmax(normalize_JSD_mean, dim=0).diag() # weight for each hyperedge (column in hypergraph)

        ## local Hypergraph convolutions
        # local Hypergraph convolutions for numerical feature
        for i, LocalHGConv_time in enumerate(self.LocalHGConv_time):
            residual = x_time
            out = LocalHGConv_time(H=H, x=x_time, W=W)
            out = self.dropout(out)
            x_time = residual + out

        # local Hypergraph convolutions for news feature
        for i, LocalHGConv_news in enumerate(self.LocalHGConv_news):
            residual = x_news
            out = LocalHGConv_news(H=H, x=x_news, W=W)
            out = self.dropout(out)
            x_news = residual + out

        # reshape back to the original shape
        return x_time.reshape(self.N, self.T, self.dim), x_news.reshape(self.N, self.T, self.dim)

class GlobalHypergraph(nn.Module):
    def __init__(self, N, T, dim, E2, num_heads_MHSA=2, num_Global_HGConv=1, dropout=0.1, eps = 1e-6):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        ## intra modality
        # numerical feature, Multihead self-Attention
        self.MHSA_time = nn.MultiheadAttention(
            embed_dim=T*dim,
            num_heads=num_heads_MHSA,
            dropout=dropout,
        )
        # news, Multihead self-Attention
        self.MHSA_news = nn.MultiheadAttention(
            embed_dim=T*dim,
            num_heads=num_heads_MHSA,
            dropout=dropout,
        )

        # inter modality
        # numerical feature to news, Multihead cross-Attention
        self.MHSA_time2news = nn.MultiheadAttention(
            embed_dim=T*dim,
            num_heads=num_heads_MHSA,
            dropout=dropout,
        )
        # news to numerical feature, Multihead cross-Attention
        self.MHSA_news2time = nn.MultiheadAttention(
            embed_dim=T*dim,
            num_heads=num_heads_MHSA,
            dropout=dropout,
        )

        ## Global sub-hyperedge embedding network, FFN, E2 is the number of global hyperedges
        self.H_time = nn.Sequential(
            nn.Linear(N, E2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E2, E2),
            )

        self.H_news = nn.Sequential(
            nn.Linear(N, E2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E2, E2),
            )

        self.H_time2news = nn.Sequential(
            nn.Linear(N, E2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E2, E2),
            )

        self.H_news2time = nn.Sequential(
            nn.Linear(N, E2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E2, E2),
            )

        ## Global hyperedge fusion network
        self.fusion = nn.Sequential(
            nn.Linear(4*E2, E2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(E2, E2),
            )


        # Hypergraph convolution layers
        self.GlobalHGConv_time = nn.ModuleList([
            HypergraphConvolution(T*dim)
            for _ in range(num_Global_HGConv)
        ])

        self.GlobalHGConv_news = nn.ModuleList([
            HypergraphConvolution(T*dim)
            for _ in range(num_Global_HGConv)
        ])

        self.N = N
        self.T = T
        self.dim = dim
        self.eps = eps



    def forward(self, x_time, x_news):
        x_time = x_time.flatten(start_dim=1) # [N, T*dim]

        x_news = x_news.flatten(start_dim=1) # [N, T*dim]

        ## intra-modality
        # time to time
        _, attn_weights_time = self.MHSA_time(x_time,x_time,x_time)
        H_time = self.H_time(attn_weights_time)
        mean = H_time.mean(dim=0, keepdim=True)    # [1, E2]
        std  = H_time.std(dim=0, keepdim=True)     # [1, E2]
        H_time = (H_time - mean) / (std + self.eps)    # 避免除0
        H_time = nnF.softmax(H_time, dim=0)   # [N, E2]

        # news to news
        _, attn_weights_news = self.MHSA_news(x_news,x_news,x_news)
        H_news = self.H_news(attn_weights_news)
        mean = H_news.mean(dim=0, keepdim=True)    # [1, E2]
        std  = H_news.std(dim=0, keepdim=True)     # [1, E2]
        H_news = (H_news - mean) / (std + self.eps)    # 避免除0
        H_news = nnF.softmax(H_news, dim=0)   # [N, E2]

        ## inter-modal
        # time to news
        _, attn_weights_time2news = self.MHSA_time2news(x_time,x_news,x_news)
        H_time2news = self.H_time2news(attn_weights_time2news)
        mean = H_time2news.mean(dim=0, keepdim=True)    # [1, E2]
        std  = H_time2news.std(dim=0, keepdim=True)     # [1, E2]
        H_time2news = (H_time2news - mean) / (std + self.eps)    # 避免除0
        H_time2news = nnF.softmax(H_time2news, dim=0)   # [N, E2]

        # news to time
        _, attn_weights_news2time = self.MHSA_news2time(x_news, x_time, x_time)
        H_news2time = self.H_news2time(attn_weights_news2time)
        mean = H_news2time.mean(dim=0, keepdim=True)    # [1, E2]
        std  = H_news2time.std(dim=0, keepdim=True)     # [1, E2]
        H_news2time = (H_news2time - mean) / (std + self.eps)    # 避免除0
        H_news2time = nnF.softmax(H_news2time, dim=0)   # [N, E2]

        # global hypergraph fusion
        H = self.fusion(torch.cat([H_time, H_news, H_time2news, H_news2time], dim=1)) # Shape: [N*T, 4*E2] -> [N*T, E2]
        mean = H.mean(dim=0, keepdim=True)
        std  = H.std(dim=0, keepdim=True)
        H = (H - mean) / (std + self.eps)
        H = nnF.softmax(H, dim=0)
        # Jensen-Shannon Divergence weighting mechanism for each hyperedge (column in hypergraph)
        JSD = jensen_shannon_divergence(H)
        JSD_mean = JSD.mean(dim=0)
        # Z-score Normalization and create diagonal weight matrix
        normalize_JSD_mean = (JSD_mean - JSD_mean.mean()) / (JSD_mean.std() + self.eps)
        W = nnF.softmax(normalize_JSD_mean, dim=0).diag() # weight for each hyperedge (column in hypergraph)

        # Apply hypergraph convolutions
        for i, GlobalHGConv_time in enumerate(self.GlobalHGConv_time):
            residual = x_time
            out = GlobalHGConv_time(H=H, x=x_time, W=W)
            out = self.dropout(out)
            x_time = residual + out

        for i, GlobalHGConv_news in enumerate(self.GlobalHGConv_news):
            residual = x_news
            out = GlobalHGConv_news(H=H, x=x_news, W=W)
            out = self.dropout(out)
            x_news = residual + out

        return x_time.reshape(self.N, self.T, self.dim), x_news.reshape(self.N, self.T, self.dim), H


class DualHGNN(nn.Module):
    def __init__(self, N, T, dim, E1, E2, num_Local_HGConv=1,
                 num_heads_MHSA=2, num_Global_HGConv=1, dropout=0.1, eps=1e-6):
        super(DualHGNN, self).__init__()
        self.LocalHGNN = LocalHypergraph(N=N, T=T, dim=dim, E1=E1, num_Local_HGConv=num_Local_HGConv, dropout=dropout, eps=eps)
        self.GlobalHGNN = GlobalHypergraph(N=N, T=T, dim=dim, E2=E2, num_heads_MHSA=num_heads_MHSA, num_Global_HGConv=num_Global_HGConv, dropout=dropout, eps=eps)

    def forward(self, x_time, x_news):
        x_time, x_news = self.LocalHGNN(x_time, x_news) # local hypergraph for cross/multimodal Alignment/Fusion
        x_time, x_news, H_glob = self.GlobalHGNN(x_time, x_news) # global hypergraph for cross/multimodal Alignment/Fusion
        return x_time, x_news, H_glob