from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data, news, timestamps, T, d):
        """
        Args:
            numerical data: Stock price/feature data [N, T_total, F]
            news: News embeddings [N, T_total, news_dim]
            timestamps: Timestamp embeddings [T_total, timestamp_dim]
            T: Lookback window size
            d: Prediction horizon (predict d days after lookback window)
        """
        super().__init__()
        self.data = data
        self.news = news
        self.timestamps = timestamps
        self.T = T
        self.d = d  # Prediction horizon
        # Adjust max_len to account for prediction horizon
        # We need T days for lookback and d-1 additional days for prediction
        self.max_len = data.shape[1] - T - d + 1

    def __len__(self):
        return max(0, self.max_len)

    def __getitem__(self, idx):
        # numerical data (lookback window)
        X = self.data[:, idx:idx + self.T, :]  # [N, T, Feat_dim], Feat: close,
        news = self.news[:, idx:idx + self.T, :]  # [N, T, llm_dim]
        timestamps = self.timestamps[idx:idx + self.T, :]  # [T, timestamp_dim]

        # calculate label - predict d days ahead
        # Compare price at day (idx + T + d - 1) with last day of lookback window
        label = (self.data[:, idx + self.T + self.d - 1, 0] - X[:, -1, 0]) > 0
        label = label.long()

        return X, news, timestamps, label