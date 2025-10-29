import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import os


class PortfolioTradingStrategy:
    """
    Portfolio Trading Strategy with d-day rebalancing and configurable rising threshold
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 transaction_cost_rate: float = 0.0025,
                 k_stocks: Optional[int] = None,
                 p_ratio: Optional[float] = None,
                 q_stop_loss: float = 0.5,
                 r_rising_ratio: float = 1.0,
                 risk_free_rate: float = 0.02,
                 d: int = 1,
                 rising_threshold: float = 0.5,
                 data_name: str = ""):
        """
        Initialize the trading strategy

        Args:
            initial_capital: Starting capital (default: 1,000,000)
            transaction_cost_rate: Cost per trade as percentage (default: 0.25%)
            k_stocks: Fixed number of stocks to hold (if None, use p_ratio)
            p_ratio: Proportion of stocks to hold (0 < p <= 1)
            q_stop_loss: Stop loss threshold ratio (0 < q < 1)
            r_rising_ratio: Ratio of rising stocks to buy when stop-loss not triggered (0 <= r <= 1)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            d: Prediction horizon - rebalance every d days (default: 1)
            rising_threshold: Probability threshold for classifying a stock as rising (default: 0.5)
            data_name: Name of the dataset for labeling outputs
        """
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.k_stocks = k_stocks
        self.p_ratio = p_ratio
        self.q_stop_loss = q_stop_loss
        self.r_rising_ratio = r_rising_ratio
        self.risk_free_rate = risk_free_rate
        self.d = d
        self.rising_threshold = rising_threshold
        self.data_name = data_name

        # Validate parameters
        if k_stocks is None and p_ratio is None:
            raise ValueError("Either k_stocks or p_ratio must be specified")
        if p_ratio is not None and not (0 < p_ratio <= 1):
            raise ValueError("p_ratio must be 0 < p_ratio <= 1")
        if not (0 < q_stop_loss < 1):
            raise ValueError("q_stop_loss must be 0 < q_stop_loss < 1")
        if not (0 <= r_rising_ratio <= 1):
            raise ValueError("r_rising_ratio must be 0 <= r_rising_ratio <= 1")
        if d < 1:
            raise ValueError("d must be >= 1")
        if not (0 <= rising_threshold <= 1):
            raise ValueError("rising_threshold must be 0 <= rising_threshold <= 1")

        # Initialize tracking variables
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking variables for a new backtest"""
        self.portfolio_values = []
        self.daily_returns = []
        self.cumulative_returns = []
        self.positions = []
        self.cash_history = []
        self.transaction_costs = []
        self.transaction_counts = []
        self.daily_holdings = []
        self.rebalancing_days = []

        # Tracking for precision metric
        self.precision_tracking = []
        self.selected_stocks_history = []
        self.actual_price_changes = []

    def calculate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities using softmax"""
        return torch.softmax(logits, dim=-1)

    def select_stocks(self, rise_probs: torch.Tensor, n_stocks: int) -> Tuple[List[int], int]:
        """Select stocks based on rise probability and stop-loss mechanism"""
        # Determine target number of stocks
        if self.k_stocks is not None:
            top_k = min(self.k_stocks, n_stocks)
        else:
            top_k = int(self.p_ratio * n_stocks)

        # Find stocks predicted to rise using configurable threshold
        rising_stocks = torch.where(rise_probs > self.rising_threshold)[0]
        m_rising = len(rising_stocks)

        # Apply stop-loss mechanism
        stop_loss_threshold = int(top_k * self.q_stop_loss)

        if m_rising >= top_k:
            # Buy all top K stocks
            top_k_indices = torch.topk(rise_probs, top_k).indices
            selected_stocks = top_k_indices.tolist()
        elif m_rising >= stop_loss_threshold:
            # Buy only M*r stocks predicted to rise
            num_to_select = int(m_rising * self.r_rising_ratio)
            if num_to_select > 0:
                rising_probs = rise_probs[rising_stocks]
                top_indices = torch.topk(rising_probs, num_to_select).indices
                selected_stocks = rising_stocks[top_indices].tolist()
                top_k = num_to_select
            else:
                selected_stocks = []
                top_k = 0
        else:
            # Stop buying today
            selected_stocks = []
            top_k = 0

        return selected_stocks, top_k

    def calculate_transaction_cost(self, value: float) -> float:
        """Calculate transaction cost for a given trade value"""
        return value * self.transaction_cost_rate

    def execute_trades(self,
                       current_holdings: Dict[int, float],
                       selected_stocks: List[int],
                       stock_prices: np.ndarray,
                       cash: float) -> Tuple[Dict[int, float], float, float, int]:
        """Execute trades to transition from current to target holdings"""
        new_holdings = {}
        total_cost = 0
        num_transactions = 0

        # Step 1: Liquidate positions not in target
        for stock_idx, shares in current_holdings.items():
            if stock_idx not in selected_stocks:
                sale_value = shares * stock_prices[stock_idx]
                transaction_cost = self.calculate_transaction_cost(sale_value)
                cash += sale_value - transaction_cost
                total_cost += transaction_cost
                num_transactions += 1
            else:
                new_holdings[stock_idx] = shares

        # Calculate portfolio value including cash
        portfolio_value = cash
        for stock_idx, shares in new_holdings.items():
            portfolio_value += shares * stock_prices[stock_idx]

        # Step 2: Calculate target allocation (equal value)
        if len(selected_stocks) > 0:
            target_value_per_stock = portfolio_value / (1 + self.transaction_cost_rate) / len(selected_stocks)

            # Step 3: Rebalance existing positions and buy new ones
            for stock_idx in selected_stocks:
                current_value = new_holdings.get(stock_idx, 0) * stock_prices[stock_idx]

                if abs(current_value - target_value_per_stock) > 1e-6:
                    if current_value < target_value_per_stock:
                        buy_value = target_value_per_stock - current_value
                        shares_to_buy = buy_value / stock_prices[stock_idx]
                        transaction_cost = self.calculate_transaction_cost(buy_value)
                        new_holdings[stock_idx] = new_holdings.get(stock_idx, 0) + shares_to_buy
                        cash -= buy_value + transaction_cost
                        total_cost += transaction_cost
                        num_transactions += 1
                    else:
                        sell_value = current_value - target_value_per_stock
                        shares_to_sell = sell_value / stock_prices[stock_idx]
                        transaction_cost = self.calculate_transaction_cost(sell_value)
                        new_holdings[stock_idx] -= shares_to_sell
                        cash += sell_value - transaction_cost
                        total_cost += transaction_cost
                        num_transactions += 1

        return new_holdings, cash, total_cost, num_transactions

    def calculate_portfolio_value(self, holdings: Dict[int, float], stock_prices: np.ndarray, cash: float) -> float:
        """Calculate total portfolio value given holdings and prices"""
        portfolio_value = cash
        for stock_idx, shares in holdings.items():
            portfolio_value += shares * stock_prices[stock_idx]
        return portfolio_value

    def run_backtest(self,
                     logits: torch.Tensor,
                     close_prices: torch.Tensor) -> Dict[str, float]:
        """Run the complete backtest with d-day rebalancing"""
        # Reset tracking
        self.reset_tracking()

        # Convert tensors to numpy
        logits_np = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
        prices_np = close_prices.cpu().numpy() if isinstance(close_prices, torch.Tensor) else close_prices

        n_stocks, t_days = logits_np.shape[0], logits_np.shape[1]

        # Verify price data
        max_trading_days = t_days // self.d * self.d + self.d - 1
        if prices_np.shape[1] < max_trading_days + 1:
            raise ValueError(f"Insufficient price data. Need at least {max_trading_days + 1} days, got {prices_np.shape[1]}")

        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}

        # Initial portfolio value
        self.portfolio_values.append(self.initial_capital)
        self.cash_history.append(cash)

        # Run trading cycle
        for day in range(max_trading_days):
            current_prices = prices_np[:, day]

            # Check if it's a rebalancing day
            if day % self.d == 0:
                self.rebalancing_days.append(day)

                # Get predictions
                day_logits = torch.tensor(logits_np[:, day, :])
                day_probs = self.calculate_probabilities(day_logits)
                rise_probs = day_probs[:, 1]

                # Select stocks
                selected_stocks, _ = self.select_stocks(rise_probs, n_stocks)
                self.selected_stocks_history.append(selected_stocks)

                # Calculate actual price changes for precision
                if len(selected_stocks) > 0 and day + self.d < prices_np.shape[1]:
                    rebalancing_prices = prices_np[:, day]
                    future_prices = prices_np[:, day + self.d]
                    actual_rises = []
                    for stock_idx in selected_stocks:
                        price_change = (future_prices[stock_idx] - rebalancing_prices[stock_idx]) / rebalancing_prices[stock_idx]
                        actual_rises.append(price_change > 0)
                    self.actual_price_changes.append(actual_rises)
                else:
                    self.actual_price_changes.append([])

                # Execute trades
                holdings, cash, transaction_cost, num_trades = self.execute_trades(
                    holdings, selected_stocks, current_prices, cash
                )

                self.transaction_costs.append(transaction_cost)
                self.transaction_counts.append(num_trades)
            else:
                self.transaction_costs.append(0)
                self.transaction_counts.append(0)

            # Record daily holdings
            self.daily_holdings.append(list(holdings.keys()))

            # Calculate end-of-day portfolio value
            next_prices = prices_np[:, day + 1]
            portfolio_value = self.calculate_portfolio_value(holdings, next_prices, cash)

            # Record metrics
            self.portfolio_values.append(portfolio_value)
            self.cash_history.append(cash)

            # Calculate daily return
            daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)

            # Calculate cumulative return
            cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.cumulative_returns.append(cumulative_return)

            self.positions.append(holdings.copy())

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()

        # Add rebalancing-specific metrics
        metrics['rebalancing_frequency'] = self.d
        metrics['num_rebalancing_days'] = len(self.rebalancing_days)
        metrics['rising_threshold'] = self.rising_threshold
        metrics['prediction_precision'] = self.calculate_precision()

        return metrics

    def calculate_precision(self) -> float:
        """Calculate the precision of stock selection predictions"""
        total_predictions = 0
        correct_predictions = 0

        for selected_stocks, actual_outcomes in zip(self.selected_stocks_history, self.actual_price_changes):
            if len(selected_stocks) > 0 and len(actual_outcomes) > 0:
                correct_predictions += sum(actual_outcomes)
                total_predictions += len(actual_outcomes)

        if total_predictions > 0:
            precision = correct_predictions / total_predictions
        else:
            precision = 0.0

        return precision

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics"""
        metrics = {}

        trading_days = len(self.daily_returns)
        if trading_days <= 0:
            raise ValueError("No trading days to calculate metrics")

        # Basic metrics
        metrics['final_portfolio_value'] = self.portfolio_values[-1]
        metrics['cumulative_return'] = self.cumulative_returns[-1]

        # Annual return (252 trading days per year)
        # Use geometric mean for proper annualization
        periods = 252  # periods per year
        metrics['annual_return'] = ((1 + metrics['cumulative_return']) ** (periods / trading_days)) - 1

        # Risk metrics
        returns_array = np.array(self.daily_returns)


        std_daily = np.std(returns_array, ddof=1)

        ## Volatility
        # metrics['volatility'] = std_daily * np.sqrt(periods)

        # Daily risk-free rate (compound)
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / periods) - 1.0

        # Sharpe Ratio
        excess_daily = returns_array - rf_daily
        mean_excess_daily = np.mean(excess_daily)
        if std_daily > 0:
            metrics['sharpe_ratio'] = (mean_excess_daily / std_daily * np.sqrt(periods))
        else:
            # print("Standard deviation is zero")
            metrics['sharpe_ratio'] = np.nan

        # # Sortino Ratio calculation
        # downside = np.minimum(excess_daily, 0.0)
        # downside_std_daily = np.sqrt(np.mean(downside ** 2))
        # if downside_std_daily > 0:
        #     metrics['sortino_ratio'] = mean_excess_daily / downside_std_daily * np.sqrt(periods)
        # else:
        #     print("Downside deviation is zero")
        #     metrics['sortino_ratio'] = np.nan

        # Maximum Drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0.0
        for value in self.portfolio_values[1:]:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        metrics['max_drawdown'] = max_drawdown

        # Calmar Ratio
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / metrics['max_drawdown']
        else:
            print("Max drawdown is zero")
            metrics['calmar_ratio'] = np.nan

        # # Win Rate
        # metrics['win_rate'] = float(np.mean(returns_array >= 0.0))

        # # Trading activity metrics
        # metrics['total_transactions'] = sum(self.transaction_counts)
        # metrics['total_transaction_costs'] = sum(self.transaction_costs)
        # metrics['avg_daily_transactions'] = metrics['total_transactions'] / trading_days

        return metrics

    def plot_results(self, data_name, save_path: Optional[str] = None):
        """Create visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Backtesting Results on {data_name} (d={self.d} days rebalancing)', fontsize=16)

        # 1. Portfolio Value Over Time
        ax = axes[0]
        ax.plot(self.portfolio_values, linewidth=2, color='blue')
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.legend()

        # 2. Daily Returns Distribution
        ax = axes[1]
        if self.daily_returns:
            ax.hist([r * 100 for r in self.daily_returns], bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax.set_title('Daily Returns Distribution')
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            mean_return = np.mean(self.daily_returns) * 100
            ax.axvline(x=mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.2f}%')
            ax.legend()

        # 3. Drawdown Chart
        ax = axes[2]
        drawdowns = []
        peak = self.portfolio_values[0]
        for value in self.portfolio_values:
            peak = max(peak, value)
            drawdown = (value - peak) / peak * 100
            drawdowns.append(drawdown)

        ax.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        ax.plot(drawdowns, color='darkred', linewidth=1)
        ax.set_title('Drawdown Over Time')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"Backtesting_result_{data_name}.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            base, ext = save_path.rsplit('.', 1)
            new_save_path = f"{base}_{data_name}.{ext}"
            plt.savefig(new_save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def print_performance_summary(self, metrics: Dict[str, float]):
        """Print a formatted summary of performance metrics"""
        print("\n" + "=" * 60)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print(f"Rebalancing Frequency: Every {self.d} day(s)")
        print("=" * 60)

        print("\nRETURN METRICS:")
        print(f"  Initial Capital:         {self.initial_capital:,.2f}")
        print(f"  Final Portfolio Value:   {metrics['final_portfolio_value']:,.2f}")
        print(f"  Cumulative Return:       {metrics['cumulative_return'] * 100:.2f}%")
        print(f"  Annual Return (AR):      {metrics['annual_return'] * 100:.2f}%")

        print("\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:            {metrics['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio:            {metrics['calmar_ratio']:.3f}")

        print("\nRISK METRICS:")
        print(f"  Maximum Drawdown:        {metrics['max_drawdown'] * 100:.2f}%")

        print("\nPREDICTION METRICS:")
        print(f"  Prediction Precision:    {metrics.get('prediction_precision', 0) * 100:.2f}%")

        print("\nTRADING ACTIVITY:")
        print(f"  Rebalancing Days:        {metrics['num_rebalancing_days']}")

        print("=" * 60)