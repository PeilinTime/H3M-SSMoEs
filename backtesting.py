import torch
from tqdm import tqdm
from torcheval.metrics.functional import binary_accuracy
from Dataset import StockDataset
from model import Model
from backtesting_tool import PortfolioTradingStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def generate_close_price(data_path, T):
    data = torch.load(data_path)

    total_date = data.shape[1]
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)

    test_data = data[:, valid_cutoff:]
    test_close_price = test_data[:, T - 1:, 0]

    return test_close_price


def generate_logits(data_path, news_path, timestamps_path, weight_path, model_params, T, d):
    """
    Generate logits by loading model weights and running inference

    Args:
        data_path: Path to market data
        news_path: Path to news embeddings
        timestamps_path: Path to timestamps
        weight_path: Path to saved model weights
        model_params: Dictionary containing model parameters
        T: Lookback window
        d: Prediction horizon

    Returns:
        test_logits: Generated logits for test set
        test_labels: True labels for test set
        test_accuracy: Accuracy on test set
    """
    print("Loading data...")
    data = torch.load(data_path).to(device)
    news_embeddings = torch.load(news_path).to(device)
    timestamps = torch.load(timestamps_path).to(device)

    # Data parameters
    N, total_date, F = data.shape
    news_dim = news_embeddings.shape[-1]
    timestamps_dim = timestamps.shape[-1]

    # Data split
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)

    test_data = data[:, valid_cutoff:]
    test_news = news_embeddings[:, valid_cutoff:]
    test_timestamps = timestamps[valid_cutoff:]

    # Standardize test data using Z-Score
    epsilon = 1e-6
    test_data_mean = test_data.mean(dim=1, keepdim=True)
    test_data_std = test_data.std(dim=1, keepdim=True)
    test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

    # Create dataset
    test_dataset = StockDataset(test_data, test_news, test_timestamps, T, d)

    # Calculate the number of trading days
    T_test = len(test_dataset)

    # Initialize model with parameters
    model = Model(
        time_dim=F,
        news_dim=news_dim,
        timestamps_dim=timestamps_dim,
        llm_hidden_size=model_params['llm_hidden_size'],
        N=N,
        T=T,
        dim=model_params['dim'],
        E1=model_params['E'],
        E2=model_params['E'],
        llm_ckp_dir=model_params['llm_ckp_dir'],
        device=device,
        num_Local_HGConv=model_params.get('num_Local_HGConv', 1),
        num_heads_MHSA=model_params.get('num_heads_MHSA', 2),
        num_Global_HGConv=model_params.get('num_Global_HGConv', 1),
        market_dim=model_params.get('market_dim', 16),
        num_market_experts=model_params['num_market_experts'],
        top_k_market=model_params['top_k'],
        num_industry_experts=model_params['num_industry_experts'],
        top_k_industry=model_params['top_k'],
        style_dim=model_params.get('style_dim', 16),
        alpha=model_params.get('aux_loss_weight', 1e-1),
        beta=model_params.get('aux_loss_weight', 1e-1),
        dropout=model_params.get('dropout', 0.1),
        noisy=False
    ).to(device)

    # Load model weights
    print(f"Loading model weights from {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Initialize output tensors
    test_logits = torch.zeros(N, T_test, 2, device=device)
    test_labels = torch.zeros(N, T_test, dtype=torch.long, device=device)

    # Generate predictions
    with torch.no_grad():
        for idx in tqdm(range(T_test), desc="Generating outputs"):
            X, news, ts, y = test_dataset[idx]
            X = X.to(device)
            news = news.to(device)
            ts = ts.to(device)
            y = y.to(device)

            outputs, _, _ = model(X, news, ts)
            test_logits[:, idx, :] = outputs
            test_labels[:, idx] = y

    # Calculate accuracy
    logits_flat = test_logits.reshape(-1, 2)
    labels_flat = test_labels.reshape(-1)
    preds = logits_flat.argmax(dim=1)
    test_accuracy = binary_accuracy(preds, labels_flat).item()

    return test_logits.cpu(), test_labels.cpu(), test_accuracy


class Backtest:
    """Main class for running backtests with model weights"""

    def __init__(self, data_name):
        self.data_name = data_name
        self.device = device

        # Dataset-specific configurations
        self.configs = {
            "DJIA": {
                "data_path": 'djia_alpha158_alpha360.pt',
                "news_path": 'djia_news_embeddings.pt',
                "timestamps_path": 'timestamps_embedding.pt',
                "weight_path": 'DJIA_weight.pth',
                "T": 20,
                "d": 10,
                "p_ratio": 1,
                "q_stop_loss": 0.05,
                "r_rising_ratio": 0.05,
                "model_params": {
                    'dim': 256,
                    'E': 64,
                    'num_market_experts': 3,
                    'num_industry_experts': 10,
                    'top_k': 2,
                    'aux_loss_weight': 1e-1,
                    'llm_hidden_size': 2048,
                    'llm_ckp_dir': 'llama-3.2-1b',
                    'num_Local_HGConv': 1,
                    'num_heads_MHSA': 2,
                    'num_Global_HGConv': 1,
                    'market_dim': 16,
                    'style_dim': 16,
                    'dropout': 0.1
                }
            },
            "NASDAQ100": {
                "data_path": 'nas100_alpha158_alpha360.pt',
                "news_path": 'nas100_news_embeddings.pt',
                "timestamps_path": 'timestamps_embedding.pt',
                "weight_path": 'NASDAQ100_weight.pth',
                "T": 20,
                "d": 10,
                "p_ratio": 1,
                "q_stop_loss": 0.05,
                "r_rising_ratio": 0.15,
                "model_params": {
                    'dim': 256,
                    'E': 32,
                    'num_market_experts': 5,
                    'num_industry_experts': 6,
                    'top_k': 2,
                    'aux_loss_weight': 1e-1,
                    'llm_hidden_size': 2048,
                    'llm_ckp_dir': 'llama-3.2-1b',
                    'num_Local_HGConv': 1,
                    'num_heads_MHSA': 2,
                    'num_Global_HGConv': 1,
                    'market_dim': 16,
                    'style_dim': 16,
                    'dropout': 0.1
                }
            },
            "S&P100": {
                "data_path": 'sp100_alpha158_alpha360.pt',
                "news_path": 'sp100_news_embeddings.pt',
                "timestamps_path": 'timestamps_embedding.pt',
                "weight_path": 'SP100_weight.pth',
                "T": 20,
                "d": 10,
                "p_ratio": 1,
                "q_stop_loss": 0.65,
                "r_rising_ratio": 0.25,
                "model_params": {
                    'dim': 256,
                    'E': 32,
                    'num_market_experts': 3,
                    'num_industry_experts': 8,
                    'top_k': 2,
                    'aux_loss_weight': 1e-1,
                    'llm_hidden_size': 2048,
                    'llm_ckp_dir': 'llama-3.2-1b',
                    'num_Local_HGConv': 1,
                    'num_heads_MHSA': 2,
                    'num_Global_HGConv': 1,
                    'market_dim': 16,
                    'style_dim': 16,
                    'dropout': 0.1
                }
            }
        }

        self.config = self.configs.get(data_name, self.configs["DJIA"])

    def run_backtest(self):
        """Run complete backtest pipeline"""
        print(f"\n{'=' * 60}")
        print(f"RUNNING BACKTESTING FOR {self.data_name}")
        print(f"{'=' * 60}")

        # Generate logits from weights
        test_logits, test_labels, test_accuracy = generate_logits(
            data_path=self.config['data_path'],
            news_path=self.config['news_path'],
            timestamps_path=self.config['timestamps_path'],
            weight_path=self.config['weight_path'],
            model_params=self.config['model_params'],
            T=self.config['T'],
            d=self.config['d']
        )

        # print(f"\nClassification Accuracy: {test_accuracy:.4f}")

        # Generate close prices
        test_close_prices = generate_close_price(self.config['data_path'], self.config['T'])

        # Create trading strategy
        strategy = PortfolioTradingStrategy(
            initial_capital=1_000_000,
            transaction_cost_rate=0.0025,
            p_ratio=self.config['p_ratio'],
            q_stop_loss=self.config['q_stop_loss'],
            r_rising_ratio=self.config['r_rising_ratio'],
            risk_free_rate=0.02,
            d=self.config['d'],
            rising_threshold=0.5,
            data_name=self.data_name
        )

        # Run backtest
        metrics = strategy.run_backtest(test_logits, test_close_prices)

        # Add accuracy to metrics
        metrics['accuracy'] = test_accuracy

        # Print performance summary with accuracy
        self.print_performance_summary_with_accuracy(strategy, metrics)

        # Plot results
        strategy.plot_results(self.data_name)

        return metrics, strategy

    def print_performance_summary_with_accuracy(self, strategy, metrics):

        print("\n" + "=" * 60)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print(f"Rebalancing Frequency: Every {strategy.d} day(s)")
        print("=" * 60)

        print("\nRETURN METRICS:")
        print(f"  Initial Capital:         {strategy.initial_capital:,.2f}")
        print(f"  Final Portfolio Value:   {metrics['final_portfolio_value']:,.2f}")
        print(f"  Cumulative Return:       {metrics['cumulative_return'] * 100:.2f}%")
        print(f"  Annual Return (AR):      {metrics['annual_return'] * 100:.2f}%")

        print("\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:             {metrics['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio:             {metrics['calmar_ratio']:.3f}")

        print("\nRISK METRICS:")
        print(f"  Maximum Drawdown:         {metrics['max_drawdown'] * 100:.2f}%")

        print("\nPrediction METRICS:")
        print(f"  Accuracy (ACC):           {metrics['accuracy'] * 100:.2f}%")
        print(f"  Prediction Precision:     {metrics.get('prediction_precision', 0) * 100:.2f}%")



def run_all_datasets():
    """Run backtests for all datasets and create comparison"""
    results = {}

    datasets = ["DJIA", "NASDAQ100", "S&P100"]

    for data_name in datasets:

        print(f"\n{'#' * 80}")
        print(f"PROCESSING {data_name} DATASET")
        print(f"{'#' * 80}")

        backtest = Backtest(data_name)
        metrics, _ = backtest.run_backtest()
        results[data_name] = metrics

    # Print comparison table
    if results:
        print("\n" + "=" * 80)
        print("PERFORMANCES OF ALL DATASETS")
        print("=" * 80)

        # Header with accuracy added
        print(f"{'Dataset':<12} {'ACC (%)':<10} {'PRE (%)':<10} {'AR (%)':<10} {'SR':<10} {'CR':<10} {'MDD (%)':<10}")
        print("-" * 72)

        # Data rows
        for data_name, metrics in results.items():
            print(f"{data_name:<12} "
                  f"{metrics['accuracy'] * 100:<10.2f} "
                  f"{metrics['prediction_precision'] * 100:<10.2f} "
                  f"{metrics['annual_return'] * 100:<10.2f} "
                  f"{metrics['sharpe_ratio']:<10.3f} "
                  f"{metrics['calmar_ratio']:<10.3f} "
                  f"{metrics['max_drawdown'] * 100:<10.2f} ")


    return results


# Main execution
if __name__ == "__main__":
    # Run all three datasets
    all_results = run_all_datasets()

    ### Or run single dataset:

    # # DJIA:
    # backtest = Backtest("DJIA")
    # metrics, strategy = backtest.run_backtest()

    # # NASDAQ100:
    # backtest = Backtest("NASDAQ100")
    # metrics, strategy = backtest.run_backtest()

    # # S&P100:
    # backtest = Backtest("S&P100")
    # metrics, strategy = backtest.run_backtest()

