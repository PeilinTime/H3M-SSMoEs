import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from tool import Metrics


def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_aux_loss_market = 0
    total_aux_loss_industry = 0

    output_list = []
    y_list = []

    # Store n_stocks for later use in metrics calculation
    n_stocks = None

    progress_bar = tqdm(dataloader, desc='Training')
    for X_batch, news_batch, timestamps_batch, y_batch in progress_bar:  # X_batch: [B, N, T, F], y_batch:[B, N]
        batch_size = X_batch.size(0)

        # Get n_stocks from the batch shape (only need to do this once)
        if n_stocks is None:
            n_stocks = X_batch.shape[1]  # X_batch shape is [B, N, T, F], so n_stocks = N

        # Zero gradients once per batch
        optimizer.zero_grad()

        # Process all samples in batch
        batch_outputs = []
        batch_labels = []
        total_batch_loss = 0
        total_batch_cls_loss = 0
        total_batch_aux_loss_market = 0
        total_batch_aux_loss_industry = 0

        # Forward pass with autocast for all samples in batch
        with autocast():
            for b in range(batch_size):
                X = X_batch[b]  # X: [N, T, F]
                news = news_batch[b]  # news: [N, T, llm_dim]
                timestamps = timestamps_batch[b]  # [N, T, llm_dim]
                y = y_batch[b]  # y: [N]

                # Store labels for metrics calculation
                batch_labels.append(y)

                # Forward pass
                outputs, aux_loss_market, aux_loss_industry = model(X, news, timestamps)  # model's output
                batch_outputs.append(outputs)

                # Accumulate loss (without backward yet)
                cls_loss = criterion(outputs, y)
                total_batch_cls_loss += cls_loss
                total_batch_aux_loss_market += aux_loss_market
                total_batch_aux_loss_industry += aux_loss_industry
                total_batch_loss += cls_loss + aux_loss_market + aux_loss_industry

            # Single backward pass for the accumulated loss
            avg_batch_loss = total_batch_loss / batch_size
            avg_batch_cls_loss = total_batch_cls_loss / batch_size
            avg_batch_aux_loss_market = total_batch_aux_loss_market / batch_size
            avg_batch_aux_loss_industry = total_batch_aux_loss_industry / batch_size

            scaler.scale(avg_batch_loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Single optimizer step for the entire batch
        scaler.step(optimizer)
        scaler.update()

        # Calculate metrics for the entire batch
        with torch.no_grad():
            batch_outputs_cat = torch.cat(batch_outputs, dim=0)
            batch_labels_cat = torch.cat(batch_labels, dim=0)

            # Save for overall metrics calculation
            output_list.append(batch_outputs_cat.detach())
            y_list.append(batch_labels_cat.detach())

            # Calculate batch metrics for progress bar
            batch_metrics = Metrics.calculate_metrics(batch_outputs_cat, batch_labels_cat, n_stocks=n_stocks)

            # Update progress bar with batch metrics
            progress_bar.set_postfix(
                {'loss': avg_batch_loss.item(), 'cls_loss': avg_batch_cls_loss.item(),
                 'aux_loss_market': avg_batch_aux_loss_market.item(),
                 'aux_loss_industry': avg_batch_aux_loss_industry.item(),
                 'accuracy': batch_metrics['accuracy'],
                 })

        total_loss += avg_batch_loss.item()
        total_cls_loss += avg_batch_cls_loss.item()
        total_aux_loss_market += avg_batch_aux_loss_market.item()
        total_aux_loss_industry += avg_batch_aux_loss_industry.item()

    # Calculate final metrics across all batches
    with torch.no_grad():
        all_outputs = torch.cat(output_list, dim=0)
        all_labels = torch.cat(y_list, dim=0)
        final_metrics = Metrics.calculate_metrics(all_outputs, all_labels, n_stocks=n_stocks)
        final_metrics['loss'] = total_loss / len(dataloader)
        final_metrics['cls_loss'] = total_cls_loss / len(dataloader)
        final_metrics['aux_loss_market'] = total_aux_loss_market / len(dataloader)
        final_metrics['aux_loss_industry'] = total_aux_loss_industry / len(dataloader)

    return final_metrics


def validate(model, dataloader, criterion, mode):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_aux_loss_market = 0
    total_aux_loss_industry = 0
    output_list = []
    y_list = []

    # Store n_stocks for later use in metrics calculation
    n_stocks = None

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=mode)
        for X_batch, news_batch, timestamps_batch, y_batch in progress_bar:  # X_batch: [B, N, T, F], y_batch:[B, N]
            batch_size = X_batch.size(0)

            # Get n_stocks from the batch shape (only need to do this once)
            if n_stocks is None:
                n_stocks = X_batch.shape[1]  # X_batch shape is [B, N, T, F]

            batch_outputs = []
            batch_labels = []

            # Process each sample in the batch
            total_batch_loss = 0
            total_batch_cls_loss = 0
            total_batch_aux_loss_market = 0
            total_batch_aux_loss_industry = 0

            for b in range(batch_size):
                X = X_batch[b]  # X: [N, T, F]
                news = news_batch[b]  # [N, T, llm_dim]
                timestamps = timestamps_batch[b]  # [N, T, llm_dim]
                y = y_batch[b]  # y: [N]

                # Store labels for metrics calculation
                batch_labels.append(y)

                # Forward pass
                outputs, aux_loss_market, aux_loss_industry = model(X, news, timestamps)  # model's output
                batch_outputs.append(outputs)

                # Compute loss
                cls_loss = criterion(outputs, y)
                total_batch_cls_loss += cls_loss
                total_batch_aux_loss_market += aux_loss_market
                total_batch_aux_loss_industry += aux_loss_industry
                total_batch_loss += cls_loss + aux_loss_market + aux_loss_industry

            # Average batch loss
            avg_batch_loss = total_batch_loss / batch_size
            total_loss += avg_batch_loss.item()
            avg_batch_cls_loss = total_batch_cls_loss / batch_size
            total_cls_loss += avg_batch_cls_loss.item()
            avg_batch_aux_loss_market = total_batch_aux_loss_market / batch_size
            total_aux_loss_market += avg_batch_aux_loss_market.item()
            avg_batch_aux_loss_industry = total_batch_aux_loss_industry / batch_size
            total_aux_loss_industry += avg_batch_aux_loss_industry.item()

            # Concatenate batch outputs and labels for overall metrics
            batch_outputs_cat = torch.cat(batch_outputs, dim=0)
            batch_labels_cat = torch.cat(batch_labels, dim=0)

            output_list.append(batch_outputs_cat)
            y_list.append(batch_labels_cat)

            # Calculate batch metrics for progress bar
            batch_metrics = Metrics.calculate_metrics(batch_outputs_cat, batch_labels_cat, n_stocks=n_stocks)

            # Update progress bar with batch metrics
            progress_bar.set_postfix(
                {'loss': avg_batch_loss.item(), 'cls_loss': avg_batch_cls_loss.item(),
                 'aux_loss_market': avg_batch_aux_loss_market.item(),
                 'aux_loss_industry': avg_batch_aux_loss_industry.item(),
                 'accuracy': batch_metrics['accuracy'],
                 })

    # Calculate final metrics across all batches with per-time top-k
    all_outputs = torch.cat(output_list, dim=0)
    all_labels = torch.cat(y_list, dim=0)
    final_metrics = Metrics.calculate_metrics(all_outputs, all_labels, n_stocks=n_stocks)
    final_metrics['loss'] = total_loss / len(dataloader)
    final_metrics['cls_loss'] = total_cls_loss / len(dataloader)
    final_metrics['aux_loss_market'] = total_aux_loss_market / len(dataloader)
    final_metrics['aux_loss_industry'] = total_aux_loss_industry / len(dataloader)

    return final_metrics