import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split

# Set a device to run the model on
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout layer with dropout probability of 0.5
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout layer with dropout probability of 0.5
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define your loss function
loss_function = nn.BCEWithLogitsLoss()

# Define your optimizer
def get_model(lr):
    model = NeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    return model, optimizer

# Define a function to train the model for one epoch
def train(train_loader, model, optimizer, loss_function):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# Define a function to evaluate the model
def evaluate(val_loader, model, loss_function):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


# Define a function to compute accuracy
def compute_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  # Convert logits to binary predictions
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total


# Load the data
data = pd.concat([pd.read_csv('/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'), 
                  pd.read_csv('/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'),
                  pd.read_csv('/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')],
                  axis=0, ignore_index=True)

data.drop(data.columns[0], axis=1, inplace=True)
data[data.columns[-1]] = data[data.columns[-1]].astype(float)

# Assuming the last column is the label/target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Perform train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define the number of epochs
epochs = range(50)  # Adjust the number of epochs as needed
batch_size = 32
learning_rate = 0.001

final_test_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model, optimizer = get_model(learning_rate)
    
    for epoch in epochs:  # Adjust the number of epochs as needed
        train_loss = train(train_loader, model, optimizer, loss_function)
        val_loss = evaluate(val_loader, model, loss_function)
        val_accuracy = compute_accuracy(val_loader, model)  # Calculate validation accuracy
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    
    # Test on the validation set
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loss = evaluate(val_loader, model, loss_function)
    print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
    
    final_test_losses.append(val_loss)

# Calculate and print the average validation loss across all folds
average_validation_loss = sum(final_test_losses) / len(final_test_losses)
print(f'Average Validation Loss across all Folds: {average_validation_loss:.4f}')

# Now you can use the test set for final validation
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss = evaluate(test_loader, model, loss_function)
print(f'Final Test Loss: {test_loss:.4f}')
test_accuracy = compute_accuracy(test_loader, model)
print(f'Final Test Accuracy: {test_accuracy:.4f}')
