# download_data.py

import torchvision
import torchvision.transforms as transforms

def download_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Downloading MNIST...")
    _ = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    _ = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print("MNIST download complete.")

def download_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Downloading CIFAR-10...")
    _ = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    _ = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("CIFAR-10 download complete.")

if __name__ == "__main__":
    # download_mnist()
    download_cifar10()
	# torchvision.datasets.MNIST("data/mnist", train=True, download=True)
