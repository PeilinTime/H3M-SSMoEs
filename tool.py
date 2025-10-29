from torcheval.metrics.functional import binary_accuracy
import torch


class Metrics:
    @staticmethod
    def calculate_topk_precision_per_time(probabilities, targets, k, n_stocks):
        """
        Calculate per-time precision for top-k stocks based on predicted probabilities.

        Args:
            probabilities: Tensor of shape [N*T] with predicted probabilities for positive class
            targets: Tensor of shape [N*T] with true labels (0 or 1)
            k: Number of top stocks to select (or fraction if k < 1)
            n_stocks: Number of stocks (N)

        Returns:
            Average top-k precision across all time points as a float
        """
        # Reshape to [T, N] where T is number of time points, N is number of stocks
        total_samples = len(probabilities)
        n_timepoints = total_samples // n_stocks # T

        if total_samples % n_stocks != 0:
            print(f"Total samples {total_samples} not divisible by n_stocks {n_stocks}!")
            raise
            # return 0.0

        # Reshape probabilities and targets to [T, N]
        probs_reshaped = probabilities.view(n_timepoints, n_stocks)
        targets_reshaped = targets.view(n_timepoints, n_stocks)

        # Calculate top-k precision for each time point
        precisions = []

        for t in range(n_timepoints):
            probs_t = probs_reshaped[t]  # [N]
            targets_t = targets_reshaped[t]  # [N]

            # Handle percentage-based k
            if k < 1:  # Treat as percentage
                k_actual = max(1, int(n_stocks * k))
            else:
                k_actual = min(k, n_stocks)  # Ensure k doesn't exceed number of stocks

            # Get indices of top k stocks by probability at time t
            _, top_k_indices = torch.topk(probs_t, k_actual)

            # Get the actual labels for top k predicted stocks
            top_k_targets = targets_t[top_k_indices]

            # Calculate precision: fraction of top k that are actually positive
            topk_precision = top_k_targets.float().mean()
            precisions.append(topk_precision.item())

        if not precisions:
            raise RuntimeError("No precision values were computed!")

        # Return average precision across all time points
        # return sum(precisions) / len(precisions) if precisions else 0.0
        return sum(precisions) / len(precisions)

    @staticmethod
    def calculate_metrics(logits, targets, n_stocks=None) -> dict:
        """
        Calculate metrics using torchmetrics

        Args:
            logits: Model output logits [N*T, 2]
            targets: True labels [N*T]
            n_stocks: Number of stocks (N).
        """
        preds = logits.argmax(dim=1)

        # accuracy (calculated globally)
        accuracy = binary_accuracy(preds, targets)

        return {
            'accuracy': accuracy.item(),
        }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def split_and_standardize_data(data, news_embeddings, timestamps, train_ratio=0.7, valid_ratio=0.1, epsilon=1e-6):
    """
    Split data into train/valid/test sets and standardize each separately.

    Parameters:
    -----------
    data : torch.Tensor
        Input data tensor of shape (features, time_steps)
    T : int
        Lookback window (default: 10)
    train_ratio : float
        Proportion of data for training (default: 0.7)
    valid_ratio : float
        Proportion of data for validation (default: 0.1)
    epsilon : float
        Small value to avoid division by zero (default: 1e-6)

    Returns:
    --------
    train_data : torch.Tensor
        Standardized training data
    valid_data : torch.Tensor
        Standardized validation data
    test_data : torch.Tensor
        Standardized test data
    """
    # Data split
    total_date = data.shape[1]
    train_cutoff = int(total_date * train_ratio)
    valid_cutoff = train_cutoff + int(total_date * valid_ratio)

    train_data = data[:, :train_cutoff]
    valid_data = data[:, train_cutoff:valid_cutoff]
    test_data = data[:, valid_cutoff:]

    # Standardize training data
    train_data_mean = train_data.mean(dim=1, keepdim=True)
    train_data_std = train_data.std(dim=1, keepdim=True)
    train_data = (train_data - train_data_mean) / (train_data_std + epsilon)

    # Standardize validation data
    valid_data_mean = valid_data.mean(dim=1, keepdim=True)
    valid_data_std = valid_data.std(dim=1, keepdim=True)
    valid_data = (valid_data - valid_data_mean) / (valid_data_std + epsilon)

    # Standardize test data
    test_data_mean = test_data.mean(dim=1, keepdim=True)
    test_data_std = test_data.std(dim=1, keepdim=True)
    test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

    train_news = news_embeddings[:, :train_cutoff]
    valid_news = news_embeddings[:, train_cutoff:valid_cutoff]
    test_news = news_embeddings[:, valid_cutoff:]

    train_timestamps = timestamps[:train_cutoff]
    valid_timestamps = timestamps[train_cutoff:valid_cutoff]
    test_timestamps = timestamps[valid_cutoff:]

    return train_data, train_news, train_timestamps, valid_data, valid_news, valid_timestamps, test_data, test_news, test_timestamps


def check_tensor_validity(tensor):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan and has_inf:
        print("Tensor contains both NaN and Inf values")
    elif has_nan:
        print("Tensor contains NaN values")
    elif has_inf:
        print("Tensor contains Inf values")
    else:
        print("Tensor contains valid values only")