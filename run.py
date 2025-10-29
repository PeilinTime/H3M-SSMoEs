import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from Dataset import StockDataset
from model import Model
from tool import EarlyStopping, split_and_standardize_data
from train import train_epoch, validate
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
    """Main training function for model training"""

    # Training configuration
    train_config = {
        'batch_size': 2,
        'epochs': 40,
        'patience': 40,
        'epsilon': 1e-6,
        'train_ratio': 0.7,
        'valid_ratio': 0.1
    }

    # Model configuration
    model_config = {
        'llm_ckp_dir': 'llama-3.2-1b',
        'llm_hidden_size': 2048,
        'num_Local_HGConv': 1,
        'num_heads_MHSA': 2,
        'num_Global_HGConv': 1,
        'market_dim': 16,
        'style_dim': 16,
        'dropout': 0.1,
        'noisy': False
    }



    # Dataset-specific hyperparameter configurations
    ## For DJIA:
    hyperparams = {
        'dim': 256,
        'E': 64,
        'num_market_experts': 3,
        'num_industry_experts': 10,
        'T': 20,
        'd': 10,
        'lr': 1e-4,
        'aux_loss_weight': 1e-1,
        'top_k': 2
    }
    data_path = 'djia_alpha158_alpha360.pt'
    news_path = 'djia_news_embeddings.pt'


    # ## For NASDAQ 100:
    # hyperparams = {
    #     'dim': 256,
    #     'E': 32,
    #     'num_market_experts': 5,
    #     'num_industry_experts': 6,
    #     'T': 20,
    #     'd': 10,
    #     'lr': 1e-4,
    #     'aux_loss_weight': 1e-1,
    #     'top_k': 2
    # }
    # data_path = 'nas100_alpha158_alpha360.pt'
    # news_path = 'nas100_news_embeddings.pt'



    # ## For S&P100:
    # hyperparams = {
    #     'dim': 256,
    #     'E': 32,
    #     'num_market_experts': 3,
    #     'num_industry_experts': 8,
    #     'T': 20,
    #     'd': 10,
    #     'lr': 1e-4,
    #     'aux_loss_weight': 1e-1,
    #     'top_k': 2
    # }
    # data_path = 'sp100_alpha158_alpha360.pt'
    # news_path = 'sp100_news_embeddings.pt'


    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"training_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    timestamps_path = 'timestamps_embedding.pt'
    data = torch.load(data_path).to(device)
    news_embeddings = torch.load(news_path).to(device)
    timestamps = torch.load(timestamps_path).to(device)

    # Split data
    train_data, train_news, train_timestamps, valid_data, valid_news, valid_timestamps, test_data, test_news, test_timestamps = split_and_standardize_data(
        data, news_embeddings, timestamps,
        train_ratio=train_config['train_ratio'],
        valid_ratio=train_config['valid_ratio'],
        epsilon=train_config['epsilon']
    )

    N, time_dim = data.shape[0], data.shape[-1]
    news_dim = news_embeddings.shape[-1]
    timestamps_dim = timestamps.shape[-1]

    # Create datasets and dataloaders
    train_dataset = StockDataset(train_data, train_news, train_timestamps, hyperparams['T'], hyperparams['d'])
    valid_dataset = StockDataset(valid_data, valid_news, valid_timestamps, hyperparams['T'], hyperparams['d'])
    test_dataset = StockDataset(test_data, test_news, test_timestamps, hyperparams['T'], hyperparams['d'])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # Initialize model
    model = Model(
        time_dim=time_dim,
        news_dim=news_dim,
        timestamps_dim=timestamps_dim,
        llm_ckp_dir=model_config['llm_ckp_dir'],
        llm_hidden_size=model_config['llm_hidden_size'],
        N=N,
        T=hyperparams['T'],
        dim=hyperparams['dim'],
        E1=hyperparams['E'],
        E2=hyperparams['E'],
        device=device,
        num_Local_HGConv=model_config['num_Local_HGConv'],
        num_heads_MHSA=model_config['num_heads_MHSA'],
        num_Global_HGConv=model_config['num_Global_HGConv'],
        market_dim=model_config['market_dim'],
        num_market_experts=hyperparams['num_market_experts'],
        top_k_market=hyperparams['top_k'],
        num_industry_experts=hyperparams['num_industry_experts'],
        top_k_industry=hyperparams['top_k'],
        style_dim=model_config['style_dim'],
        alpha=hyperparams['aux_loss_weight'],
        beta=hyperparams['aux_loss_weight'],
        noisy=model_config['noisy'],
        dropout=model_config['dropout'],
        eps=train_config['epsilon']
    ).to(device)

    # Initialize training components
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=0.05)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=train_config['patience'])

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * train_config['epochs']
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Paths for saving
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    csv_filename = os.path.join(output_dir, 'training_metrics.csv')

    # Initialize tracking variables
    best_val_acc = 0.
    best_test_metrics = None
    best_val_metrics = None
    best_train_metrics = None
    best_epoch = 0

    # Record training history
    history = {
        'epoch': [],
        'train_loss': [], 'train_cls_loss': [], 'train_aux_loss_market': [], 'train_aux_loss_industry': [],
        'train_accuracy': [],

        'val_loss': [], 'val_cls_loss': [], 'val_aux_loss_market': [], 'val_aux_loss_industry': [],
        'val_accuracy': [],

        'test_loss': [], 'test_cls_loss': [], 'test_aux_loss_market': [], 'test_aux_loss_industry': [],
        'test_accuracy': [],
    }

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    try:
        for epoch in range(train_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{train_config['epochs']}")

            # Training
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler)

            # Validation and testing
            with torch.no_grad():
                val_metrics = validate(model, valid_loader, criterion, mode='validating')
                test_metrics = validate(model, test_loader, criterion, mode='testing')

            # Update learning rate
            lr_scheduler.step()

            # Early stopping check
            early_stopping(val_metrics['accuracy'])

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

                # Store metrics from the best epoch
                best_test_metrics = test_metrics.copy()
                best_val_metrics = val_metrics.copy()
                best_train_metrics = train_metrics.copy()
                best_epoch = epoch + 1

            # Update history
            history['epoch'].append(epoch + 1)
            for phase in ['train', 'val', 'test']:
                metrics = locals()[f'{phase}_metrics']
                for metric_name in ['loss', 'cls_loss', 'aux_loss_market', 'aux_loss_industry',
                                    'accuracy',
                                    ]:
                    history[f'{phase}_{metric_name}'].append(metrics[metric_name])

            # Save metrics to CSV after each epoch
            df = pd.DataFrame(history)
            df.to_csv(csv_filename, index=False)

            # Print summary metrics
            print(f"Train - Total loss: {train_metrics['loss']:.4f}, cls_loss: {train_metrics['cls_loss']:.4f}, aux_loss_market: {train_metrics['aux_loss_market']:.4f}, aux_loss_industry: {train_metrics['aux_loss_industry']:.4f}, accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Valid - Total loss: {val_metrics['loss']:.4f}, cls_loss: {val_metrics['cls_loss']:.4f}, aux_loss_market: {val_metrics['aux_loss_market']:.4f}, aux_loss_industry: {val_metrics['aux_loss_industry']:.4f}, accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Test - Total loss: {test_metrics['loss']:.4f}, cls_loss: {test_metrics['cls_loss']:.4f}, aux_loss_market: {test_metrics['aux_loss_market']:.4f}, aux_loss_industry: {test_metrics['aux_loss_industry']:.4f}, accuracy: {test_metrics['accuracy']:.4f}")

            if early_stopping.early_stop:
                print("\nEarly stopping triggered!")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest model achieved at epoch {best_epoch}:")
    print(f"\nFiles saved to: {output_dir}")
    print(f"  Best model: {best_model_path}")
    print(f"  Training metrics: {csv_filename}")

    return {
        'best_epoch': best_epoch,
        'best_test_metrics': best_test_metrics,
        'best_val_metrics': best_val_metrics,
        'best_train_metrics': best_train_metrics,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    results = main()