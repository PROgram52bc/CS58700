import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import DigitCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model and weights
digit_net = DigitCNN().to(device)
digit_net.load_state_dict(torch.load("net0_digit.pth"))
digit_net.eval()

# Load MNIST test data
transform = transforms.ToTensor()
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)
testloader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Evaluate accuracy
correct = 0
total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = digit_net(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Digit classifier accuracy: {100 * correct / total:.2f}%")

