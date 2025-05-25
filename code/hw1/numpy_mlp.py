import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

from hw1_learning_curves import save_learning_curve

def load_fashion_mnist():
    print("Loading Fashion-MNIST...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, cache=True)
    X = X.to_numpy().astype('float32')
    y = y.to_numpy().astype('int32')
    X = X / 255.0  # Normalize to [0,1]
    X = X * 2 - 1   # Scale to [-1,1]
    return train_test_split(X, y, test_size=0.2, random_state=42)
class NumpyResMLP:
    def __init__(self, input_size, hidden_size, num_classes=10):
        # He initialization for weights, zeros for biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_size + 1, hidden_size) * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden_size + 1, hidden_size) * np.sqrt(2 / hidden_size)
        self.W3 = np.random.randn(hidden_size + 1, num_classes) * np.sqrt(2 / hidden_size)

    def forward(self, X):
        batch_size = X.shape[0]

        # Layer 1: Fully connected + ReLU
        self.X_ = np.hstack((X, np.ones((batch_size, 1))))
        self.Z1 = self.X_ @ self.W1
        self.H1 = np.maximum(0, self.Z1)  

        # Layer 2: Residual connection
        self.H1_ = np.hstack((self.H1, np.ones((batch_size, 1))))
        self.Z2 = self.H1_ @ self.W2
        self.H2 = np.maximum(0, self.Z2 + self.H1)

        # Output layer: Log-Softmax
        self.H2_ = np.hstack((self.H2, np.ones((batch_size, 1))))
        self.Z3 = self.H2_ @ self.W3

        # Log-Softmax with numerical stability
        Z3_max = np.max(self.Z3, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(self.Z3 - Z3_max), axis=-1, keepdims=True))
        self.Y_hat = self.Z3 - Z3_max - log_sum_exp

        return self.Y_hat

    def backward(self, X, y, lr=0.01):
        batch_size = X.shape[0]

        # One-hot encoding of ground-truth labels
        y_onehot = np.eye(self.num_classes)[y]

        # Step 1: Compute gradient of Log-Softmax + NLL loss
        softmax = np.exp(self.Y_hat)
        dL_dz3 = (softmax - y_onehot) / batch_size  

        # Step 2: Gradient for output layer weights W3
        dL_dw3 = self.H2_.T @ dL_dz3  

        # Step 3: Backprop through residual connection
        dL_da2 = dL_dz3 @ self.W3[:-1, :].T  
        da2_dsum = ((self.Z2 + self.H1) > 0).astype(float)  
        dL_dsum = dL_da2 * da2_dsum  

        # Step 4: Backprop through layer 2
        dL_dz2 = dL_dsum  
        dL_da1_res = dL_dsum  
        dL_da1_dz2 = dL_dz2 @ self.W2[:-1, :]  

        # Gradient for W2
        dL_dw2 = self.H1_.T @ dL_dz2  

        # Step 5: Combine residual gradient paths and backprop through ReLU
        dL_da1 = dL_da1_res + dL_da1_dz2  
        dL_dz1 = dL_da1 * (self.Z1 > 0).astype(float)  

        # Step 6: Gradient for first layer weights W1
        dL_dw1 = self.X_.T @ dL_dz1  

        # Step 7: Update weights using SGD
        self.W1 -= lr * dL_dw1
        self.W2 -= lr * dL_dw2
        self.W3 -= lr * dL_dw3

    def predict(self, X):
        log_probs = self.forward(X)
        return np.argmax(log_probs, axis=1)

def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST MLP with NumPy')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=512)
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_fashion_mnist()
    
    # Initialize model
    model = NumpyResMLP(input_size=784, hidden_size=args.hidden_size)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    n_batches = len(X_train) // args.batch_size

    print(f"Total train size {len(X_train)}, Total test size {len(X_test)}")
    
    for epoch in range(args.epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{args.epochs}'):
            start_idx = i * args.batch_size
            end_idx = start_idx + args.batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Your code here
            # Forward + backward, also compute and book-keep loss
            # Think about the gradient of NLL, do I need log softmax for backward?
            log_probs = model.forward(X_batch)
            batch_loss = -np.mean(log_probs[np.arange(len(y_batch)), y_batch]) # the loss function
            epoch_loss += batch_loss
            model.backward(X_batch, y_batch, lr=0.05)
            
            preds = np.argmax(log_probs, axis=1)
            correct += np.sum(preds == y_batch)
            total += len(y_batch)
        
        # Calculate epoch metrics
        train_loss = epoch_loss / n_batches
        train_acc = 100 * correct / total
        
        # Evaluate on test set
        test_preds = model.predict(X_test)
        test_acc = 100 * np.mean(test_preds == y_test)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('numpy_results.png')
    plt.show()


def main_learning_curve():
    # Your code here
    # Implement the training code that plots the learning curve. 
    # You can use the save_learning_curve function from hw1_learning_curves.py.
    raise NotImplementedError


if __name__ == '__main__':
    main()

    # NOTE: Comment out the above main() and uncomment the below main_learning_curve() to run the learning curve code for Q5.4
    # main_learning_curve()
