import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


# Set a device to run the model on
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.cuda.is_available()
    else 'cpu'
)

print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        # calls constuctor from superclass
        super().__init__()
        self.flatten = nn.Flatten()
        # build sequence of operations of linear layers followed by relu activation functions
        # change activation function and layers to experiment with the outcome 
        self.linear_relu_stack = nn.Sequential(
            # 28 * 28 refers to number of pixels in a given image
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
    
    def forward(self, x):
        # flatten 2D image data to form input vector into a neural network
        x = self.flatten(x)
        # feed data to linear_relu_stack defined in __init__
        logits = self.linear_relu_stack(x)
        return logits



training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

model = NeuralNetwork()

# Define hyperparameters for training
learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss



epochs = 100
epoch_list = np.arange(0, epochs, 1)
loss_list = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = training_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    loss_list.append(loss)
print("Done!")

# Save the model
torch.save(model.state_dict(), "ML/image_calssification_test.pth")
print("Saved PyTorch Model State to image_calssification_test.pth")

# plot loss over time
plt.plot(epoch_list, loss_list, label="Loss over time", color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.legend()
plt.savefig('ML/loss_over_time.png')
plt.show()
