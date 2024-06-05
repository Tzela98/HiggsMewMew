import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# train_model function trains the model for one epoch using the training data.
# returns: average loss for the epoch.


def train_model(train_loader, model, criterion, optimizer, device, clip_grad=None):
    model.train()
    running_loss = 0.0
    processed_samples = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze()

        # Ensure the outputs and labels are compatible for loss calculation
        if outputs.shape != labels.shape:
            raise ValueError(f"Shape mismatch: outputs.shape {outputs.shape} does not match labels.shape {labels.shape}")

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping if necessary
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Update statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        processed_samples += batch_size

        progress_bar.set_postfix(loss=running_loss / processed_samples)

    avg_loss = running_loss / processed_samples
    return avg_loss


# evaluate_model function evaluates the model using the validation data.
# returns: average loss, accuracy, and predictions for the validation data.


def evaluate_model(valid_loader, model, criterion, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    processed_samples = 0

    # Use tqdm for the progress bar
    progress_bar = tqdm(valid_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            all_outputs.extend((torch.sigmoid(outputs)).cpu().numpy())
            
            # Ensure outputs and labels are treated as batches
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            processed_samples += inputs.size(0)
            
            # Apply sigmoid to outputs to get predictions for binary classification
            preds = (torch.sigmoid(outputs) > threshold).float()

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update the progress bar
            progress_bar.set_postfix(loss=running_loss / processed_samples)
    
    # Compute the average loss and accuracy
    avg_loss = running_loss / processed_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_outputs, all_labels