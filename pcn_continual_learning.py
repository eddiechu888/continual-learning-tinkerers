"""
PCN Continual Learning Demo
---------------------------
This script demonstrates how Predictive Coding Networks (PCN) can mitigate catastrophic
forgetting in continual learning scenarios using Split-Fashion-MNIST dataset.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jpc import layers, inference, learning
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Check if directory exists for saving data
if not os.path.exists('data'):
    os.makedirs('data')

# Define task splits for Split-Fashion-MNIST
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to download and prepare dataset
def prepare_dataset():
    """Download and prepare Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())  # Convert to numpy for JAX
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
def evaluate(model, tasks, test_dataset):
    """Evaluate model on specific tasks and return accuracy."""
    accuracies = []
    
    for cls_ids in tasks:
        loader = make_loader(test_dataset, cls_ids, batch_size=256, shuffle=False)
        correct = 0
        total = 0
        
        for x_batch, y_batch in loader:
            # Prepare input
            x_batch = x_batch.reshape(x_batch.shape[0], -1)  # Flatten images
            
            # Forward pass (inference)
            z0 = jnp.array(x_batch)
            zs = inference.gradient_flow(model, z0, None, steps=10, lr_z=0.2)
            
            # Get predictions
            logits = zs[-1][-1]  # Last layer's output
            predictions = jnp.argmax(logits, axis=1)
            
            # Calculate accuracy
            y_batch = jnp.array(y_batch)
            correct += jnp.sum(predictions == y_batch)
            total += len(y_batch)
        
        accuracies.append(float(correct) / total if total > 0 else 0)
    
    return accuracies

def main():
    """Main function to run the continual learning experiment."""
    print("Starting PCN Continual Learning Demo...")
    
    # Prepare datasets
    print("Downloading and preparing Fashion-MNIST dataset...")
    train_dataset, test_dataset = prepare_dataset()
    
    # Create PCN model
    print("Creating PCN model...")
    model = layers.SequentialPC([
        layers.LinearPC(28*28, 256, param_scale="mup"),
        layers.ReluPC(),
        layers.LinearPC(256, 256, param_scale="mup"),
        layers.ReluPC(),
        layers.LinearPC(256, 10, param_scale="mup")
    ])
    
    # Create optimizer
    opt = learning.Adam(1e-3)  # Same learning rate for all tasks
    
    # Lists to track performance
    task_accuracies = []
    forgetting_rates = []
    
    # Training loop with continual stream
    print("Starting continual learning training...")
    for t, cls_ids in enumerate(TASKS, 1):
        print(f"\nTraining on Task {t}: Classes {[CLASS_NAMES[i] for i in cls_ids]}")
        loader = make_loader(train_dataset, cls_ids)
        
        # Training on current task
        for x_batch, y_batch in tqdm(loader, desc=f"Task {t}"):
            # Prepare input and target
            z0 = jnp.array(x_batch.reshape(x_batch.shape[0], -1))  # Flatten images
            zT = jax.nn.one_hot(jnp.array(y_batch), 10)  # One-hot encode targets
            
            # Fast loop: inference
            zs = inference.gradient_flow(model, z0, zT, steps=10, lr_z=0.2)
            
            # Slow loop: Hebbian weight update
            model, opt = learning.local_update(model, zs, opt)
        
        # Evaluate on all tasks seen so far
        current_tasks = TASKS[:t]
        accuracies = evaluate(model, current_tasks, test_dataset)
        
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
    plt.title('Task Accuracy over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot forgetting rate
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(forgetting_rates)+1), forgetting_rates, marker='o', color='red')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Forgetting Rate')
    plt.title('Forgetting Rate over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pcn_continual_learning_results.png')
    plt.show()

if __name__ == "__main__":
    main()
