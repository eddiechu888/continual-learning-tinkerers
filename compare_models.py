"""
PCN vs Backpropagation Comparison for Continual Learning
-------------------------------------------------------
This script runs both PCN and BP models on the Split-Fashion-MNIST continual learning task
and provides a side-by-side comparison of their performance.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jpc import layers, inference, learning

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
import time

# Check if directory exists for saving data
if not os.path.exists('data'):
    os.makedirs('data')

# Define task splits for Split-Fashion-MNIST
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the PyTorch MLP model
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
    # For PyTorch
    transform_torch = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # For JAX (convert to numpy)
    transform_jax = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())
    ])
    
    train_dataset_torch = FashionMNIST(root='./data', train=True, download=True, transform=transform_torch)
    test_dataset_torch = FashionMNIST(root='./data', train=False, download=True, transform=transform_torch)
    
    train_dataset_jax = FashionMNIST(root='./data', train=True, download=False, transform=transform_jax)
    test_dataset_jax = FashionMNIST(root='./data', train=False, download=False, transform=transform_jax)
    
    return (train_dataset_torch, test_dataset_torch), (train_dataset_jax, test_dataset_jax)

# Function to create data loaders for specific tasks
def make_loader_torch(dataset, cls_ids, batch_size=256, shuffle=True):
    """Create a PyTorch DataLoader for specific class IDs."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in cls_ids]
    subset = Subset(dataset, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def make_loader_jax(dataset, cls_ids, batch_size=256, shuffle=True):
    """Create a JAX-compatible DataLoader for specific class IDs."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in cls_ids]
    subset = Subset(dataset, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Function to evaluate BP model
def evaluate_bp(model, tasks, test_dataset, device):
    """Evaluate BP model on specific tasks and return accuracy."""
    model.eval()
    accuracies = []
    
    for cls_ids in tasks:
        loader = make_loader_torch(test_dataset, cls_ids, batch_size=256, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.view(x_batch.size(0), -1).to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracies.append(correct / total if total > 0 else 0)
    
    return accuracies

# Function to evaluate PCN model
def evaluate_pcn(model, tasks, test_dataset):
    """Evaluate PCN model on specific tasks and return accuracy."""
    accuracies = []
    
    for cls_ids in tasks:
        loader = make_loader_jax(test_dataset, cls_ids, batch_size=256, shuffle=False)
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

# Function to generate "dream" images from PCN model
def generate_dream_images(model, class_idx, num_images=5):
    """Generate dream images by clamping the top layer and running inference."""
    dream_images = []
    
    for _ in range(num_images):
        # Create one-hot encoded class label
        zT = jax.nn.one_hot(jnp.array([class_idx]), 10)[0]
        
        # Start with zeros at the bottom layer
        z0 = jnp.zeros((1, 28*28))
        
        # Run gradient flow to generate the dream
        zs = inference.gradient_flow(model, z0, zT.reshape(1, -1), steps=50, lr_z=0.5)
        
        # Get the generated image
        dream_image = zs[0][-1].reshape(28, 28)
        dream_images.append(dream_image)
    
    return dream_images

def main():
    """Main function to run the comparison experiment."""
    print("Starting PCN vs BP Comparison for Continual Learning...")
    
    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Downloading and preparing Fashion-MNIST dataset...")
    (train_dataset_torch, test_dataset_torch), (train_dataset_jax, test_dataset_jax) = prepare_dataset()
    
    # Create models
    print("Creating models...")
    # PCN model
    pcn_model = layers.SequentialPC([
        layers.LinearPC(28*28, 256, param_scale="mup"),
        layers.ReluPC(),
        layers.LinearPC(256, 256, param_scale="mup"),
        layers.ReluPC(),
        layers.LinearPC(256, 10, param_scale="mup")
    ])
    
    # BP model
    bp_model = MLP().to(device)
    
    # Create optimizers
    pcn_opt = learning.Adam(1e-3)
    bp_optimizer = optim.Adam(bp_model.parameters(), lr=1e-3)
    bp_criterion = nn.CrossEntropyLoss()
    
    # Lists to track performance
    pcn_task_accuracies = []
    pcn_forgetting_rates = []
    bp_task_accuracies = []
    bp_forgetting_rates = []
    
    # Training loop with continual stream
    print("Starting continual learning training...")
    for t, cls_ids in enumerate(TASKS, 1):
        print(f"\n=== Task {t}: Classes {[CLASS_NAMES[i] for i in cls_ids]} ===")
        
        # Create data loaders
        pcn_loader = make_loader_jax(train_dataset_jax, cls_ids)
        bp_loader = make_loader_torch(train_dataset_torch, cls_ids)
        
        # Train PCN model
        print("Training PCN model...")
        start_time = time.time()
        for x_batch, y_batch in tqdm(pcn_loader, desc=f"PCN Task {t}"):
            # Prepare input and target
            z0 = jnp.array(x_batch.reshape(x_batch.shape[0], -1))
            zT = jax.nn.one_hot(jnp.array(y_batch), 10)
            
            # Fast loop: inference
            zs = inference.gradient_flow(pcn_model, z0, zT, steps=10, lr_z=0.2)
            
            # Slow loop: Hebbian weight update
            pcn_model, pcn_opt = learning.local_update(pcn_model, zs, pcn_opt)
        
        pcn_train_time = time.time() - start_time
        
        # Train BP model
        print("Training BP model...")
        start_time = time.time()
        bp_model.train()
        for x_batch, y_batch in tqdm(bp_loader, desc=f"BP Task {t}"):
            # Prepare input and target
            x_batch = x_batch.view(x_batch.size(0), -1).to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            bp_optimizer.zero_grad()
            outputs = bp_model(x_batch)
            loss = bp_criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            bp_optimizer.step()
        
        bp_train_time = time.time() - start_time
        
        # Evaluate PCN model
        current_tasks = TASKS[:t]
        pcn_accuracies = evaluate_pcn(pcn_model, current_tasks, test_dataset_jax)
        
        # Evaluate BP model
        bp_accuracies = evaluate_bp(bp_model, current_tasks, test_dataset_torch, device)
        
        # Calculate forgetting (if applicable)
        if t > 1:
            pcn_forgetting = 1.0 - (sum(pcn_accuracies[:-1]) / len(pcn_accuracies[:-1]))
            bp_forgetting = 1.0 - (sum(bp_accuracies[:-1]) / len(bp_accuracies[:-1]))
        else:
            pcn_forgetting = 0.0
            bp_forgetting = 0.0
        
        # Store results
        pcn_task_accuracies.append(pcn_accuracies)
        pcn_forgetting_rates.append(pcn_forgetting)
        bp_task_accuracies.append(bp_accuracies)
        bp_forgetting_rates.append(bp_forgetting)
        
        # Print results
        print(f"\nResults after Task {t}:")
        print(f"PCN training time: {pcn_train_time:.2f}s, BP training time: {bp_train_time:.2f}s")
        
        print("\nPCN Model:")
        for i, acc in enumerate(pcn_accuracies):
            print(f"  Task {i+1} accuracy: {acc:.2%}")
        print(f"  Average accuracy: {sum(pcn_accuracies)/len(pcn_accuracies):.2%}")
        print(f"  Forgetting rate: {pcn_forgetting:.2%}")
        
        print("\nBP Model:")
        for i, acc in enumerate(bp_accuracies):
            print(f"  Task {i+1} accuracy: {acc:.2%}")
        print(f"  Average accuracy: {sum(bp_accuracies)/len(bp_accuracies):.2%}")
        print(f"  Forgetting rate: {bp_forgetting:.2%}")
        
        # Generate dream images after task 3 (optional)
        if t == 3:
            print("\nGenerating dream images from PCN model...")
            plt.figure(figsize=(15, 3))
            for i, class_idx in enumerate([0, 2, 4]):  # One from each task learned so far
                dream_images = generate_dream_images(pcn_model, class_idx)
                for j, img in enumerate(dream_images[:3]):  # Show 3 dreams per class
                    plt.subplot(1, 9, i*3 + j + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(f"{CLASS_NAMES[class_idx]}")
                    plt.axis('off')
            plt.tight_layout()
            plt.savefig('pcn_dream_images.png')
    
    # Plot final comparison results
    plt.figure(figsize=(15, 10))
    
    # Plot task accuracies for PCN
    plt.subplot(2, 2, 1)
    for i in range(len(TASKS)):
        task_acc = [accs[i] if i < len(accs) else None for accs in pcn_task_accuracies]
        plt.plot(range(1, i+2), task_acc[:i+1], marker='o', label=f'Task {i+1}')
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.title('PCN: Task Accuracy over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot task accuracies for BP
    plt.subplot(2, 2, 2)
    for i in range(len(TASKS)):
        task_acc = [accs[i] if i < len(accs) else None for accs in bp_task_accuracies]
        plt.plot(range(1, i+2), task_acc[:i+1], marker='o', label=f'Task {i+1}')
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.title('BP: Task Accuracy over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot forgetting rates
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(pcn_forgetting_rates)+1), pcn_forgetting_rates, marker='o', label='PCN')
    plt.plot(range(1, len(bp_forgetting_rates)+1), bp_forgetting_rates, marker='o', label='BP')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Forgetting Rate')
    plt.title('Forgetting Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot average accuracy
    plt.subplot(2, 2, 4)
    pcn_avg_acc = [sum(accs)/len(accs) for accs in pcn_task_accuracies]
    bp_avg_acc = [sum(accs)/len(accs) for accs in bp_task_accuracies]
    plt.plot(range(1, len(pcn_avg_acc)+1), pcn_avg_acc, marker='o', label='PCN')
    plt.plot(range(1, len(bp_avg_acc)+1), bp_avg_acc, marker='o', label='BP')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pcn_vs_bp_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
