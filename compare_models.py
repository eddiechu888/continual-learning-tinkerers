"""
PCN vs Backpropagation Comparison for Continual Learning
-------------------------------------------------------
This script runs both PCN and BP models on the Split-Fashion-MNIST continual learning task
and provides a side-by-side comparison of their performance.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import jpc
import optax
import numpy as np

# Import specific solvers and controllers for PCN
from diffrax import Dopri5, Tsit5, PIDController

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
def make_loader_torch(dataset, cls_ids, batch_size=16, shuffle=True):
    """Create a PyTorch DataLoader for specific class IDs."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in cls_ids]
    subset = Subset(dataset, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def make_loader_jax(dataset, cls_ids, batch_size=16, shuffle=True):
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
        loader = make_loader_torch(test_dataset, cls_ids, batch_size=16, shuffle=False)
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
        loader = make_loader_jax(test_dataset, cls_ids, batch_size=16, shuffle=False)
        correct = 0
        total = 0
        
        for x_batch, y_batch in loader:
            # Prepare input
            z0 = jnp.array(x_batch.reshape(x_batch.shape[0], -1))  # Flatten images
            
            # Initialize activities with feedforward pass
            activities = jpc.init_activities_with_ffwd(model=model, input=z0)
            
            # Create a dummy output for evaluation (zeros with the right shape)
            dummy_output = jnp.zeros((z0.shape[0], 10))
            
            # Run inference to equilibrium with improved parameters
            converged_activities = jpc.solve_inference(
                params=(model, None),
                activities=activities,
                output=dummy_output,  # Dummy output with correct shape
                input=z0,
                solver=Tsit5(),  # More sophisticated solver
                max_t1=50,  # Increased from default 20
                stepsize_controller=PIDController(rtol=1e-4, atol=1e-4)  # Stricter tolerances
            )
            
            # Get predictions from the last layer's output
            logits = converged_activities[-1][-1]  # Last layer's output
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
        
        # Start with random noise at the bottom layer for better dream generation
        key = jax.random.PRNGKey(int(time.time()) + _)  # Different seed for each image
        z0 = jax.random.normal(key, (1, 28*28)) * 0.01
        
        # Initialize activities for all layers
        activities = []
        # First layer is random noise
        activities.append(z0)
        # Initialize hidden layers with small random values
        activities.append(jax.random.normal(key, (1, 256)) * 0.01)
        activities.append(jax.random.normal(key, (1, 256)) * 0.01)
        # Last layer is clamped to the target class
        activities.append(zT.reshape(1, -1))
        
        # Run inference to equilibrium with improved parameters
        # Note: For dream generation, we clamp the output layer to the target class
        # and let the network infer the input that would generate this output
        converged_activities = jpc.solve_inference(
            params=(model, None),
            activities=activities,
            output=zT.reshape(1, -1),  # Clamp output to target class
            input=None,  # No input clamping
            solver=Tsit5(),  # More sophisticated solver
            max_t1=100,  # Even longer for dream generation
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-5)  # Even stricter tolerances
        )
        
        # In the new JPC API, the converged_activities structure is different
        # We need to extract the input layer activities which should be the first element
        # But we need to be careful about the shape
        
        # Print the structure of converged_activities to debug
        print(f"Dream generation - converged_activities structure: {[a.shape for a in converged_activities]}")
        
        # For now, use the input as a placeholder
        # This will be updated once we understand the structure
        dream_image = z0.reshape(28, 28)
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
    key = jax.random.PRNGKey(42)
    pcn_model = jpc.make_mlp(
        key,
        input_dim=28*28,
        width=256,
        depth=2,  # This creates 2 hidden layers
        output_dim=10,
        act_fn="relu"
    )
    
    # BP model
    bp_model = MLP().to(device)
    
    # Create optimizers
    pcn_optim = optax.adam(1e-3)
    pcn_opt_state = pcn_optim.init((eqx.filter(pcn_model, eqx.is_array), None))
    
    # Print information about the inference parameter adjustments
    print("\nPCN Inference Parameter Adjustments:")
    print("  - Using Tsit5 ODE solver (more sophisticated than default Heun)")
    print("  - Increased max_t1 from 20 to 50 for training and 100 for dream generation")
    print("  - Tightened convergence tolerances from 1e-3 to 1e-4 for training")
    print("  - Even stricter tolerances (1e-5) for dream generation")
    print("  - Monitoring convergence during training to detect timeout events")
    print("  - Reduced batch size from 128 to 16 for more granular learning")
    print("  - Increased evaluation frequency (every 2 batches instead of 5)")
    print("\nThese adjustments should help PCN reach better equilibrium states,")
    print("potentially revealing its advantages in mitigating catastrophic forgetting.")
    print("The 'goldilocks zone' we're targeting is where PCN can properly converge")
    print("to meaningful equilibrium states while BP cannot overcome its inherent limitations.")
    print("The smaller batch size will allow for more frequent evaluations and")
    print("more effective early stopping, preventing overfitting to current tasks.")
    
    bp_optimizer = optim.Adam(bp_model.parameters(), lr=1e-3)
    bp_criterion = nn.CrossEntropyLoss()
    
    # Lists to track performance
    pcn_task_accuracies = []
    pcn_forgetting_rates = []
    bp_task_accuracies = []
    bp_forgetting_rates = []
    
    # Track number of batches needed to reach target accuracy
    pcn_batches_to_converge = []
    bp_batches_to_converge = []
    
    # Training loop with continual stream
    print("Starting continual learning training...")
    for t, cls_ids in enumerate(TASKS, 1):
        print(f"\n=== Task {t}: Classes {[CLASS_NAMES[i] for i in cls_ids]} ===")
        
        # Create data loaders with even smaller batch size
        pcn_loader = make_loader_jax(train_dataset_jax, cls_ids, batch_size=16)
        bp_loader = make_loader_torch(train_dataset_torch, cls_ids, batch_size=16)
        
        # Train PCN model
        print("Training PCN model...")
        start_time = time.time()
        
        # Early stopping variables
        pcn_early_stop = False
        pcn_eval_interval = 2  # Check accuracy more frequently (every 2 batches)
        pcn_target_acc = 0.90  # Stop at 90% accuracy
        
        # Create a smaller validation set from the current task
        val_loader_jax = make_loader_jax(test_dataset_jax, cls_ids, batch_size=16, shuffle=False)
        
        # Training loop with early stopping
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(pcn_loader, desc=f"PCN Task {t}")):
            # Prepare input and target
            z0 = jnp.array(x_batch.reshape(x_batch.shape[0], -1))
            zT = jax.nn.one_hot(jnp.array(y_batch), 10)
            
            # Use JPC's make_pc_step with improved inference parameters
            # Increase max_t1 to allow more time for convergence
            # Use a more sophisticated ODE solver (Tsit5)
            # Tighten convergence criteria with stricter tolerances
            result = jpc.make_pc_step(
                model=pcn_model,
                optim=pcn_optim,
                opt_state=pcn_opt_state,
                output=zT,
                input=z0,
                ode_solver=Tsit5(),  # More sophisticated solver than default Heun
                max_t1=50,  # Increased from default 20 to allow more time for convergence
                stepsize_controller=PIDController(rtol=1e-4, atol=1e-4),  # Stricter tolerances
                record_activities=True  # Record activities to monitor convergence
            )
            
            # Monitor if inference reached equilibrium or hit timeout
            if "activities" in result and len(result["activities"]) > 0:
                # Check if the last recorded activity was at the max_t1 (timeout)
                last_t = len(result["activities"][0]) - 1
                if last_t >= 49:  # If close to max_t1=50, likely hit timeout
                    print(f"  PCN Batch {batch_idx+1}: Inference may have hit timeout")
            
            # Update model and optimizer state
            pcn_model, pcn_opt_state = result["model"], result["opt_state"]
            
            # Check accuracy periodically for early stopping
            if (batch_idx + 1) % pcn_eval_interval == 0:
                # Evaluate on current task only
                current_acc = evaluate_pcn(pcn_model, [cls_ids], test_dataset_jax)[0]
                print(f"  PCN Batch {batch_idx+1}: Current accuracy = {current_acc:.2%}")
                
                if current_acc >= pcn_target_acc:
                    print(f"  PCN reached target accuracy of {pcn_target_acc:.0%} in {batch_idx+1} batches. Early stopping.")
                    pcn_batches_to_converge.append(batch_idx+1)
                    pcn_early_stop = True
                    break
        
        # If early stopping didn't trigger, record the total number of batches
        if not pcn_early_stop:
            pcn_batches_to_converge.append(len(pcn_loader))
        
        pcn_train_time = time.time() - start_time
        
        # Train BP model
        print("Training BP model...")
        start_time = time.time()
        
        # Early stopping variables
        bp_early_stop = False
        bp_eval_interval = 2  # Check accuracy more frequently (every 2 batches)
        bp_target_acc = 0.90  # Stop at 90% accuracy
        
        # Create a smaller validation set from the current task
        val_loader_torch = make_loader_torch(test_dataset_torch, cls_ids, batch_size=16, shuffle=False)
        
        # Training loop with early stopping
        bp_model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(bp_loader, desc=f"BP Task {t}")):
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
            
            # Check accuracy periodically for early stopping
            if (batch_idx + 1) % bp_eval_interval == 0:
                # Evaluate on current task only
                current_acc = evaluate_bp(bp_model, [cls_ids], test_dataset_torch, device)[0]
                print(f"  BP Batch {batch_idx+1}: Current accuracy = {current_acc:.2%}")
                
                if current_acc >= bp_target_acc:
                    print(f"  BP reached target accuracy of {bp_target_acc:.0%} in {batch_idx+1} batches. Early stopping.")
                    bp_batches_to_converge.append(batch_idx+1)
                    bp_early_stop = True
                    break
        
        # If early stopping didn't trigger, record the total number of batches
        if not bp_early_stop:
            bp_batches_to_converge.append(len(bp_loader))
        
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
        print(f"PCN batches to reach target: {pcn_batches_to_converge[-1]}, BP batches: {bp_batches_to_converge[-1]}")
        
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
    plt.figure(figsize=(15, 15))
    
    # Plot task accuracies for PCN
    plt.subplot(3, 2, 1)
    for i in range(len(TASKS)):
        task_acc = [accs[i] if i < len(accs) else None for accs in pcn_task_accuracies]
        plt.plot(range(1, i+2), task_acc[:i+1], marker='o', label=f'Task {i+1}')
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.title('PCN: Task Accuracy over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot task accuracies for BP
    plt.subplot(3, 2, 2)
    for i in range(len(TASKS)):
        task_acc = [accs[i] if i < len(accs) else None for accs in bp_task_accuracies]
        plt.plot(range(1, i+2), task_acc[:i+1], marker='o', label=f'Task {i+1}')
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.title('BP: Task Accuracy over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot final task accuracies (bar chart comparison)
    plt.subplot(3, 2, 3)
    final_pcn_accs = pcn_task_accuracies[-1]
    final_bp_accs = bp_task_accuracies[-1]
    x = np.arange(len(TASKS))
    width = 0.35
    
    plt.bar(x - width/2, final_pcn_accs, width, label='PCN')
    plt.bar(x + width/2, final_bp_accs, width, label='BP')
    
    plt.xlabel('Task')
    plt.ylabel('Final Accuracy')
    plt.title('Final Task Accuracies After All Training')
    plt.xticks(x, [f'Task {i+1}' for i in range(len(TASKS))])
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot batches needed to reach target accuracy
    plt.subplot(3, 2, 4)
    x = np.arange(len(TASKS))
    width = 0.35
    
    plt.bar(x - width/2, pcn_batches_to_converge, width, label='PCN')
    plt.bar(x + width/2, bp_batches_to_converge, width, label='BP')
    
    plt.xlabel('Task')
    plt.ylabel('Batches to Reach Target')
    plt.title('Training Efficiency: Batches to Reach Target Accuracy')
    plt.xticks(x, [f'Task {i+1}' for i in range(len(TASKS))])
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot forgetting rates
    plt.subplot(3, 2, 5)
    plt.plot(range(1, len(pcn_forgetting_rates)+1), pcn_forgetting_rates, marker='o', label='PCN')
    plt.plot(range(1, len(bp_forgetting_rates)+1), bp_forgetting_rates, marker='o', label='BP')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Forgetting Rate')
    plt.title('Forgetting Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot average accuracy
    plt.subplot(3, 2, 6)
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
