# Reduce dataset size for minimal working test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
import pandas as pd
from datasets import Add2x2Dataset, SoftAdd2x2Dataset
from models import DigitCNN, Add2x2Model


# -------------------------------
# Step 4: Train for 1 epoch (quick test)
# -------------------------------
transform = transforms.ToTensor()
mnist = MNIST(root="./data", train=True, download=True, transform=transform)
dataset = Add2x2Dataset(mnist, which='row0', size=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

######################################
#  Train the digit recognizer first  #
######################################

# -------------------------------
# Prepare MNIST data
# -------------------------------
transform = transforms.ToTensor()
mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

digit_net = DigitCNN()
optimizer = torch.optim.Adam(digit_net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

results = []
# Train for 3 epochs
print("Training the image recognizer")
print("{:<6} {:<18} {:<10}".format("Epoch", "Acc (%)", "Time (s)"))
print("-" * 70)
for epoch in range(10):
    epoch_start = time.time()
    digit_net.train()
    for x, y in mnist_loader:
        logits = digit_net(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    digit_net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in mnist_loader:
            preds = digit_net(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    elapsed = time.time() - epoch_start
    print("{:<6} {:<18.2f} {:<10.2f}".format(epoch, acc, elapsed))

# Save weights
torch.save(digit_net.state_dict(), "net0_digit.pth")

# state_dict = torch.load("net0_digit.pth")
# digit_net.load_state_dict(state_dict)

model = Add2x2Model(digit_net)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for name, param in model.named_parameters():
    print(name, param.shape)

print("Training for add2x2")
results = []
for epoch in range(50): 
    model.train()
    epoch_start = time.time()

    for grid, labels in dataloader:
        preds = model(grid)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for grid, labels in dataloader:
            preds = model(grid)
            pred_ints = preds.round().clamp(0, 18).long()
            labels_ints = labels.long()
            correct += (pred_ints == labels_ints).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    elapsed = time.time() - epoch_start
    results.append({"Epoch": epoch, "Accuracy (%)": round(acc, 2), "Time (s)": round(elapsed, 2)})

# Display table
df = pd.DataFrame(results)
print(df.to_string(index=False))
