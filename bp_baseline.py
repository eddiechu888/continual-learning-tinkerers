"""
Backpropagation Baseline for Continual Learning
----------------------------------------------
This script implements a standard backpropagation MLP with the same architecture
as the PCN model to demonstrate catastrophic forgetting in continual learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Check if directory exists for saving data
if not os.path.exists('data'):
    os.makedirs('data')

# Define task splits for Split-Fashion-MNIST (same as PCN)
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the MLP model with the same architecture as PCN
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Function to prepare dataset
def prepare_dataset():
    """Download and prepare Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# Function to create data loaders for specific tasks
def make_loader(dataset, cls_ids, batch_size=256, shuffle=True):
    """Create a DataLoader for specific class IDs."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in cls_ids]
    subset = Subset(dataset, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Function to evaluate model on specific tasks
def evaluate(model, tasks, test_dataset, device):
    """Evaluate model on specific tasks and return accuracy."""
    model.eval()
    accuracies = []
    
    for cls_ids in tasks:
        loader = make_loader(test_dataset, cls_ids, batch_size=256, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.view(x_batch.size(0), -1).to(device)  # Flatten images
                y_batch = y_batch.to(device)
                
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracies.append(correct / total if total > 0 else 0)
    
    return accuracies

def main():
    """Main function to run the backpropagation baseline experiment."""
    print("Starting Backpropagation Baseline for Continual Learning...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Downloading and preparing Fashion-MNIST dataset...")
    train_dataset, test_dataset = prepare_dataset()
    
    # Create MLP model
    print("Creating MLP model...")
    model = MLP().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Same learning rate as PCN
    
    # Lists to track performance
    task_accuracies = []
    forgetting_rates = []
    
    # Training loop with continual stream
    print("Starting continual learning training...")
    for t, cls_ids in enumerate(TASKS, 1):
        print(f"\nTraining on Task {t}: Classes {[CLASS_NAMES[i] for i in cls_ids]}")
        loader = make_loader(train_dataset, cls_ids)
        
        # Training on current task
        model.train()
        for x_batch, y_batch in tqdm(loader, desc=f"Task {t}"):
            # Prepare input and target
            x_batch = x_batch.view(x_batch.size(0), -1).to(device)  # Flatten images
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        # Evaluate on all tasks seen so far
        current_tasks = TASKS[:t]
        accuracies = evaluate(model, current_tasks, test_dataset, device)
        
        # Calculate forgetting (if applicable)
        if t > 1:
            forgetting = 1.0 - (sum(accuracies[:-1]) / len(accuracies[:-1]))
        else:
            forgetting = 0.0
        
        # Store results
        task_accuracies.append(accuracies)
        forgetting_rates.append(forgetting)
        
        # Print results
        print(f"After Task {t}:")
        for i, acc in enumerate(accuracies):
            print(f"  Task {i+1} accuracy: {acc:.2%}")
        print(f"  Average accuracy: {sum(accuracies)/len(accuracies):.2%}")
        print(f"  Forgetting rate: {forgetting:.2%}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot task accuracies
    plt.subplot(1, 2, 1)
    for i in range(len(TASKS)):
        task_acc = [accs[i] if i < len(accs) else None for accs in task_accuracies]
        plt.plot(range(1, i+2), task_acc[:i+1], marker='o', label=f'Task {i+1}')
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.title('Task Accuracy over Time (BP)')
    plt.legend()
    plt.grid(True)
    
    # Plot forgetting rate
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(forgetting_rates)+1), forgetting_rates, marker='o', color='red')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Forgetting Rate')
    plt.title('Forgetting Rate over Time (BP)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bp_baseline_results.png')
    plt.show()

if __name__ == "__main__":
    main()
