# train_robot.py - Improved version to prevent overfitting
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from functorch import vmap, jacrev, hessian # deprecated
from torch.func import vmap, jacrev, hessian
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Training Configuration ---
class TrainingConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    REAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "output_exp_2025-07-22_12-23-07.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "data", "real_rob_f.pth")
    INPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "real_rob_i_scaler.joblib")
    OUTPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "real_rob_o_scaler.joblib")
    PLOT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "real_rob_perf.png")
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5


import torch.nn.functional as F

def filter_training_data(df, velocity_threshold=10.0, acceleration_threshold=50.0):
    """Filter out unrealistic or noisy data points"""
    vel_cols = ['tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    vel_mask = (df[vel_cols].abs() < velocity_threshold).all(axis=1)
    df_clean = df[vel_mask].copy()
    
    for traj_id, group in df_clean.groupby('trajectory'):
        if len(group) < 2:
            continue
        dt = 0.02
        for vel_col in vel_cols:
            acc_col = vel_col.replace('velocity', 'acceleration')
            df_clean.loc[group.index[1:], acc_col] = np.diff(group[vel_col].values) / dt
    
    acc_cols = [col for col in df_clean.columns if 'acceleration' in col]
    if acc_cols:
        acc_mask = (df_clean[acc_cols].abs() < acceleration_threshold).all(axis=1)
        df_clean = df_clean[acc_mask]
    
    return df_clean

def physics_informed_loss(model, inputs, targets, dt=0.02):
    """Add physics constraints to improve dynamics modeling"""
    current_pos = inputs[:, 3:6]
    current_vel = inputs[:, 6:9]
    
    predictions = model(inputs)
    pred_pos = predictions[:, :3]
    pred_vel = predictions[:, 3:6]
    
    # Position integration consistency
    expected_pos = current_pos + current_vel * dt
    position_consistency = F.mse_loss(pred_pos, expected_pos)
    
    # Velocity smoothness
    vel_change = pred_vel - current_vel
    smoothness_penalty = torch.mean(torch.norm(vel_change, dim=1))
    
    # Energy conservation
    current_energy = 0.5 * torch.sum(current_vel**2, dim=1)
    pred_energy = 0.5 * torch.sum(pred_vel**2, dim=1)
    energy_penalty = torch.mean(torch.abs(pred_energy - current_energy))
    
    return {
        'position_consistency': position_consistency,
        'smoothness': smoothness_penalty,
        'energy': energy_penalty
    }

"""A bigger, more precise neural network can lead to slower MPC convergence because its complexity creates a "jagged" and non-smooth function landscape. The MPC relies on linear approximations (the tangent slope) at each step to find the next best move. On a smooth landscape, these approximations are accurate over a large area, allowing the optimizer to take confident, large steps and converge quickly. On the jagged landscape of the bigger model, the linear approximation is only valid for a tiny, immediate area. This forces the optimizer to take very small, cautious steps and run many more iterations, dramatically slowing down the process. Essentially, for this type of optimization, the smoothness of the model is more important than its absolute precision, and the simpler model provides a much smoother, more navigable landscape for the controller to work with.
"""
class StatePredictor(nn.Module):
    """
    Enhanced Multi-Scale architecture with temporal dynamics for better rollout stability.
    Captures different time scales: short-term (high frequency), medium-term, and long-term (low frequency).
    """
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        
        # Short-term dynamics (high frequency, quick responses)
        self.short_term = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Medium-term dynamics (intermediate frequency)
        self.medium_term = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Long-term dynamics (low frequency, more stable trends)
        self.long_term = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )
        
        # Add skip connection for better gradient flow
        self.skip_connection = nn.Linear(input_dim, 64)
        
        # Combine with learned weights for better stability
        self.combiner = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 64, 128),  # +64 for skip connection
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        
        # Learnable time-scale weights (initialized to emphasize stability)
        self.time_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.2]))
        
        # Initialize weights for stability
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for better stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        # Normalize time weights
        weights = torch.softmax(self.time_weights, dim=0)
        
        # Apply time-scale weighting to emphasize different dynamics
        short_features = self.short_term(x) * weights[0]
        medium_features = self.medium_term(x) * weights[1]
        long_features = self.long_term(x) * weights[2]
        skip_features = torch.tanh(self.skip_connection(x))
        
        # Combine all temporal features
        combined = torch.cat([short_features, medium_features, long_features, skip_features], dim=-1)
        return self.combiner(combined)
    
    def stability_penalty(self):
        """Compute stability penalty for the time weights"""
        target_weights = torch.tensor([1.0, 0.5, 0.2], device=self.time_weights.device)
        weight_penalty = torch.sum(torch.abs(self.time_weights - target_weights))
        return 0.01 * weight_penalty

class RobotStateDataset(Dataset):
    """Custom PyTorch Dataset for single-step predictions."""
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RobotSequenceDataset(Dataset):
    """Custom PyTorch Dataset for sequence-based training."""
    def __init__(self, sequences_X, sequences_y):
        # sequences_X: (N, seq_len, 9), sequences_y: (N, seq_len, 6)
        self.sequences_X = torch.tensor(sequences_X, dtype=torch.float32)
        self.sequences_y = torch.tensor(sequences_y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences_X)
    
    def __getitem__(self, idx):
        return self.sequences_X[idx], self.sequences_y[idx]


def load_and_prepare_data(filepath):
    """
    Loads data from trajectories and creates pairs where X = [u_k, x_k]
    x_k = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    y = [x_k+1].
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

    # Filter noisy data
    df = filter_training_data(df)

    STATE_COLS = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    INPUT_COLS = ['volume_1', 'volume_2', 'volume_3']
    CURRENT_FEATURES = INPUT_COLS + STATE_COLS
    
    df.dropna(subset=CURRENT_FEATURES + ['T'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    X_list, y_list = [], []
    
    print("Processing trajectories...")
    for traj_id, group in df.groupby('trajectory'):
        if len(group) < 2:
            continue
            
        current_features = group[CURRENT_FEATURES].iloc[:-1].values
        next_state = group[STATE_COLS].iloc[1:].values
        
        # X = [u_k, x_k] (no dt)
        X_list.append(current_features)
        y_list.append(next_state)

    if not X_list:
        raise ValueError("Not enough data to create training pairs.")

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    print("Finished processing data.")
    return X, y


def create_sequence_dataset(filepath, sequence_length=8):
    """
    Create training sequences from actual trajectories for multi-step training.
    Returns sequences of (input, output) pairs for training on rollouts.
    """
    print(f"Loading sequence data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

    STATE_COLS = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    INPUT_COLS = ['volume_1', 'volume_2', 'volume_3']
    CURRENT_FEATURES = INPUT_COLS + STATE_COLS
    
    df.dropna(subset=CURRENT_FEATURES + ['T'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    sequences_X = []
    sequences_y = []
    
    print("Processing trajectory sequences...")
    for traj_id, group in df.groupby('trajectory'):
        if len(group) < sequence_length + 1:
            continue
            
        traj_data = group[CURRENT_FEATURES + STATE_COLS].values
        
        # Create overlapping sequences from this trajectory
        for i in range(len(traj_data) - sequence_length):
            # Input sequence: [u_k, x_k] for each step in sequence
            seq_x = traj_data[i:i+sequence_length, :9]  # First 9 columns: [u, x]
            # Output sequence: x_{k+1} for each step
            seq_y = traj_data[i+1:i+sequence_length+1, 3:9]  # State columns only
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
    
    if not sequences_X:
        raise ValueError("Not enough data to create sequence training pairs.")
    
    sequences_X = np.array(sequences_X)
    sequences_y = np.array(sequences_y)
    
    print(f"Created {len(sequences_X)} sequences of length {sequence_length}")
    print(f"Sequence input shape: {sequences_X.shape}")
    print(f"Sequence output shape: {sequences_y.shape}")
    
    return sequences_X, sequences_y


def multi_step_training_loss(model, input_scaler, output_scaler, seq_x, seq_y, device, epoch, max_epochs):
    """Enhanced multi-step training loss with adaptive horizon"""
    batch_size, seq_len, _ = seq_x.shape
    
    # Progressive horizon: start with 2, reach 15 by end of training
    current_horizon = min(2 + int(13 * epoch / max_epochs), 15)
    actual_horizon = min(current_horizon, seq_len)
    
    total_loss = 0.0
    
    for start_step in range(seq_len - actual_horizon + 1):
        current_x_and_u = seq_x[:, start_step, :].clone()
        step_losses = []
        
        for pred_step in range(actual_horizon):
            pred_next_state_scaled = model(current_x_and_u)
            
            target_step = start_step + pred_step
            if target_step < seq_len:
                target = seq_y[:, target_step, :]
                
                pred_next_state_np = pred_next_state_scaled.cpu().detach().numpy()
                pred_next_state = output_scaler.inverse_transform(pred_next_state_np)
                pred_next_state_tensor = torch.tensor(pred_next_state, dtype=torch.float32, device=device)
                
                # Exponentially weighted loss (later steps more important)
                weight = 1.5 ** pred_step
                step_loss = weight * F.mse_loss(pred_next_state_tensor, target)
                step_losses.append(step_loss)
                
                # Teacher forcing with decay probability
                teacher_forcing_prob = max(0.1, 1.0 - (pred_step * 0.15) - (epoch / max_epochs * 0.5))
                
                if pred_step < actual_horizon - 1 and target_step + 1 < seq_len:
                    if torch.rand(1).item() < teacher_forcing_prob:
                        next_state_portion = seq_y[:, target_step, :]
                    else:
                        next_state_portion = pred_next_state_tensor.detach()
                    
                    next_control = seq_x[:, target_step + 1, :3]
                    next_x_and_u = torch.cat([next_control, next_state_portion], dim=1)
                    
                    next_x_and_u_np = next_x_and_u.cpu().detach().numpy()
                    next_x_and_u_scaled = input_scaler.transform(next_x_and_u_np)
                    current_x_and_u = torch.tensor(next_x_and_u_scaled, dtype=torch.float32, device=device)
        
        if step_losses:
            total_loss += sum(step_losses) / len(step_losses)
    
    return total_loss / max(1, seq_len - actual_horizon + 1), current_horizon


def train_model_with_sequences(model, single_step_loader, sequence_loader, val_loader, 
                              num_epochs, learning_rate, input_scaler, output_scaler, device):
    """
    Enhanced training loop that combines single-step and multi-step training.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting enhanced training with multi-step loss...")
    for epoch in range(num_epochs):
        model.train()
        
        # Progressive learning rate
        if epoch < num_epochs // 3:
            lr = 1e-3
        elif epoch < 2 * num_epochs // 3:
            lr = 5e-4
        else:
            lr = 1e-4
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Adaptive multi-step weight: reach 80% by epoch 50
        multi_step_weight = min(0.8, epoch / (num_epochs * 0.5))
        single_step_weight = 1.0 - multi_step_weight
        
        # Train on single-step data
        single_step_losses = []
        for inputs, targets in single_step_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            single_loss = criterion(outputs, targets)
            
            # Add physics-informed loss
            physics_losses = physics_informed_loss(model, inputs, targets)
            physics_loss = (0.1 * physics_losses['position_consistency'] + 
                          0.05 * physics_losses['smoothness'] + 
                          0.01 * physics_losses['energy'])
            
            single_step_losses.append(single_loss + physics_loss)
        
        # Train on sequence data (multi-step)
        multi_step_losses = []
        current_horizon = 2
        for seq_inputs, seq_targets in sequence_loader:
            seq_inputs = seq_inputs.to(device)
            seq_targets = seq_targets.to(device)
            
            multi_loss, current_horizon = multi_step_training_loss(
                model, input_scaler, output_scaler, 
                seq_inputs, seq_targets, device, epoch, num_epochs
            )
            multi_step_losses.append(multi_loss)
        
        # Combine losses and do one optimization step
        total_single_loss = sum(single_step_losses) / len(single_step_losses) if single_step_losses else 0
        total_multi_loss = sum(multi_step_losses) / len(multi_step_losses) if multi_step_losses else 0
        
        combined_loss = single_step_weight * total_single_loss + multi_step_weight * total_multi_loss
        
        # Add stability penalty
        stability_loss = model.stability_penalty() if hasattr(model, 'stability_penalty') else 0
        total_loss = combined_loss + stability_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        history['train_loss'].append(total_loss.item())
        
        # Validation (single-step for simplicity)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss.item():.6f}, "
                  f"Val Loss: {val_loss:.6f}, Multi-step weight: {multi_step_weight:.3f}, "
                  f"Horizon: {current_horizon}")
    
    print("Enhanced training finished.")
    return history


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """The original single-step training loop (kept for compatibility)."""
    criterion = nn.MSELoss()  # Mean Squared Error is good for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print("Training finished.")
    return history


def plot_predictions(y_true, y_pred, save_path):
    """Generates Predicted vs. Actual plots and saves the figure to a file."""
    state_labels = [
        'Tip Position X', 'Tip Position Y', 'Tip Position Z',
        'Tip Velocity X', 'Tip Velocity Y', 'Tip Velocity Z'
    ]
    
    num_states = y_true.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(num_states):
        ax = axes[i]
        # Use a smaller subset of points for plotting if the test set is very large
        sample_size = min(len(y_true), 2000)
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        
        ax.scatter(y_true[indices, i], y_pred[indices, i], alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(state_labels[i], fontsize=14)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(pad=3.0)
    plt.suptitle("Predicted vs. Actual State Values on Test Set", fontsize=20, y=1.02)
    
    # Save the plot to a file.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")
    plt.close(fig) # Close the figure to free up memory


def plot_rollout_performance(results, save_path):
    """
    Plots the model's rollout performance (MSE) as a function of the prediction horizon.
    """
    horizons = results['horizons']
    avg_mse = results['avg_mse']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizons, avg_mse, 'bo-', label='Average MSE')
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=12)
    ax.set_ylabel("Average Mean Squared Error (MSE)", fontsize=12)
    ax.set_title("Model Prediction Error vs. Rollout Horizon", fontsize=16)
    ax.set_xticks(horizons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    
    # Use a logarithmic scale if the error grows very fast
    if max(avg_mse) / min(avg_mse) > 50:
        ax.set_yscale('log')
        ax.set_ylabel("Average Mean Squared Error (MSE) - Log Scale", fontsize=12)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nRollout performance plot saved to: {save_path}")
    plt.close(fig)


def plot_approximation_comparison(results_dict, save_path):
    """
    Plots comparison of different approximation orders in rollout performance.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    for i, (order_name, results) in enumerate(results_dict.items()):
        horizons = results['horizons']
        avg_mse = results['avg_mse']
        
        ax.plot(horizons, avg_mse, color=colors[i], marker=markers[i], 
                linewidth=2.5, markersize=8, linestyle=linestyles[i], 
                markerfacecolor='white', markeredgecolor=colors[i], 
                markeredgewidth=2, label=order_name)
    
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=14)
    ax.set_ylabel("Average Mean Squared Error (MSE)", fontsize=14)
    ax.set_title("Model Prediction Error vs. Rollout Horizon\n(Neural Network vs Approximations)", fontsize=16)
    ax.set_xticks(horizons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='best')
    
    # Use logarithmic scale if needed
    max_mse = max([max(results['avg_mse']) for results in results_dict.values()])
    min_mse = min([min(results['avg_mse']) for results in results_dict.values()])
    if max_mse / min_mse > 50:
        ax.set_yscale('log')
        ax.set_ylabel("Average Mean Squared Error (MSE) - Log Scale", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRollout performance comparison plot saved to: {save_path}")
    plt.close(fig)


class NeuralNetworkApproximator:
    """Class to handle neural network approximations for rollout evaluation"""
    
    def __init__(self, model, input_scaler, output_scaler, device):
        self.model = model.to(device).eval()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.device = device
        
        # Setup scaling tensors for PyTorch operations
        self.input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        self.input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        self.output_scale = torch.tensor(output_scaler.scale_, dtype=torch.float32, device=device)
        self.output_mean = torch.tensor(output_scaler.mean_, dtype=torch.float32, device=device)
    
    def full_pytorch_model(self, x_and_u_torch):
        """Full PyTorch model with scaling"""
        scaled_input = (x_and_u_torch - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        return scaled_output * self.output_scale + self.output_mean
    
    def first_order_approximation(self, x_and_u_torch, linearization_point):
        """First-order (linear) approximation of the neural network"""
        # Get the jacobian at the linearization point
        jac = jacrev(self.full_pytorch_model)(linearization_point)
        
        # Get the function value at linearization point
        f0 = self.full_pytorch_model(linearization_point)
        
        # Linear approximation: f(x) ≈ f(x0) + J(x0) * (x - x0)
        delta_x = x_and_u_torch - linearization_point
        return f0 + torch.matmul(jac, delta_x)
    
    def second_order_approximation(self, x_and_u_torch, linearization_point):
        """Second-order (quadratic) approximation of the neural network with numerical stability"""
        try:
            # Get function value, jacobian, and hessian at linearization point
            f0 = self.full_pytorch_model(linearization_point)
            jac = jacrev(self.full_pytorch_model)(linearization_point)
            hess = hessian(self.full_pytorch_model)(linearization_point)
            
            # Check for NaN/Inf in derivatives
            if torch.isnan(jac).any() or torch.isinf(jac).any():
                print("Warning: NaN/Inf detected in Jacobian, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            if torch.isnan(hess).any() or torch.isinf(hess).any():
                print("Warning: NaN/Inf detected in Hessian, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            delta_x = x_and_u_torch - linearization_point
            
            # Limit the step size to prevent explosion
            max_step_size = 0.1  # Limit how far we extrapolate
            delta_x_norm = torch.norm(delta_x)
            if delta_x_norm > max_step_size:
                delta_x = delta_x * (max_step_size / delta_x_norm)
            
            linear_term = torch.matmul(jac, delta_x)
            
            # For the quadratic term with regularization
            quadratic_term = torch.zeros_like(f0)
            hessian_regularization = 1e-6  # Small regularization term
            
            for i in range(f0.shape[0]):  # For each output dimension
                hess_i = hess[i]
                
                # Add regularization to make Hessian more stable
                hess_i_reg = hess_i + hessian_regularization * torch.eye(hess_i.shape[0], device=hess_i.device)
                
                # Check condition number - if too high, skip quadratic term
                try:
                    cond_num = torch.linalg.cond(hess_i_reg)
                    if cond_num > 1e8:
                        continue
                except:
                    continue
                
                quad_i = 0.5 * torch.matmul(torch.matmul(delta_x, hess_i_reg), delta_x)
                
                # Clip the quadratic term to prevent explosion
                quad_i = torch.clamp(quad_i, -1e3, 1e3)
                
                # Check for numerical issues in the quadratic term
                if torch.isnan(quad_i) or torch.isinf(quad_i):
                    continue
                
                quadratic_term[i] = quad_i
            
            result = f0 + linear_term + quadratic_term
            
            # Final clipping to prevent explosion
            result = torch.clamp(result, -1e4, 1e4)
            
            # Final check for NaN/Inf in result
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN/Inf detected in second-order result, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            return result
            
        except Exception as e:
            print(f"Warning: Exception in second-order approximation: {e}, falling back to first-order")
            return self.first_order_approximation(x_and_u_torch, linearization_point)


def evaluate_approximation_rollouts(model, X_test_orig, y_test_orig, input_scaler, output_scaler, device,
                                   approximation_order=1, horizons=[1, 5, 10, 20, 40], num_rollouts=100):
    """
    Evaluates neural network approximations on multi-horizon rollouts.
    
    Args:
        approximation_order (int): 1 for first-order, 2 for second-order approximation
        Other args same as evaluate_multi_horizon_rollouts
    """
    print(f"\n--- Evaluating {approximation_order}-order NN Approximation (Multi-Horizon Rollouts) ---")
    
    # Setup approximator
    approximator = NeuralNetworkApproximator(model, input_scaler, output_scaler, device)
    
    # X = [volumes_k (3), state_k (6)] -> state is columns 3 to 9
    num_state_dims = y_test_orig.shape[1]
    state_start_col = 3  # Assumes 3 volume inputs
    state_end_col = state_start_col + num_state_dims
    
    # Store results
    results = {'horizons': [], 'avg_mse': [], 'avg_mae': []}
    
    for horizon in horizons:
        # Skip if the horizon is longer than the available test data
        if horizon >= len(X_test_orig):
            print(f"Horizon {horizon} is too long for the test set ({len(X_test_orig)} samples). Skipping.")
            continue
        
        horizon_errors_mse = []
        horizon_errors_mae = []
        
        # Determine the possible starting points for a rollout of this length
        max_start_idx = len(X_test_orig) - horizon
        start_indices = np.random.choice(max_start_idx, size=min(num_rollouts, max_start_idx), replace=False)
        
        for start_idx in start_indices:
            # --- Perform one short rollout using approximation ---
            
            # Get the real starting state from the test set
            current_state = X_test_orig[start_idx, state_start_col:state_end_col]
            
            # For second-order approximation, recompute linearization point more frequently
            linearization_frequency = 5 if approximation_order == 2 else horizon  # Relinearize every 5 steps for 2nd order
            
            predicted_trajectory = []
            rollout_failed = False
            
            with torch.no_grad():
                for i in range(horizon):
                    # Use the actual control input from the test data
                    control_input = X_test_orig[start_idx + i, :state_start_col]
                    
                    # Combine current state and control input
                    x_and_u = np.concatenate([control_input, current_state])
                    x_and_u_torch = torch.tensor(x_and_u, dtype=torch.float32, device=device)
                    
                    # Recompute linearization point if needed
                    if i % linearization_frequency == 0:
                        linearization_point = x_and_u_torch.clone()
                    
                    # Predict next state using approximation
                    if approximation_order == 1:
                        next_state_pred = approximator.first_order_approximation(x_and_u_torch, linearization_point)
                    elif approximation_order == 2:
                        next_state_pred = approximator.second_order_approximation(x_and_u_torch, linearization_point)
                    else:
                        raise ValueError(f"Unsupported approximation order: {approximation_order}")
                    
                    next_state_pred_np = next_state_pred.cpu().numpy()
                    
                    # Check for numerical issues
                    if np.any(np.isnan(next_state_pred_np)) or np.any(np.isinf(next_state_pred_np)):
                        rollout_failed = True
                        break
                    
                    predicted_trajectory.append(next_state_pred_np)
                    current_state = next_state_pred_np
            
            # Only compute metrics if the rollout was successful
            if not rollout_failed and len(predicted_trajectory) == horizon:
                # Compare the predicted trajectory to the ground truth
                predicted_trajectory = np.array(predicted_trajectory)
                ground_truth_trajectory = y_test_orig[start_idx : start_idx + horizon]
                
                try:
                    horizon_errors_mse.append(mean_squared_error(ground_truth_trajectory, predicted_trajectory))
                    horizon_errors_mae.append(mean_absolute_error(ground_truth_trajectory, predicted_trajectory))
                except Exception as e:
                    print(f"Warning: Error computing metrics for rollout starting at {start_idx}: {e}")
        
        # Only compute averages if we have valid results
        if len(horizon_errors_mse) > 0:
            avg_mse = np.mean(horizon_errors_mse)
            avg_mae = np.mean(horizon_errors_mae)
            
            results['horizons'].append(horizon)
            results['avg_mse'].append(avg_mse)
            results['avg_mae'].append(avg_mae)
            
            print(f"Horizon: {horizon:3d} steps -> Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f} ({len(horizon_errors_mse)} successful rollouts)")
        else:
            print(f"Warning: No successful rollouts for horizon {horizon} steps")
    
    return results


def evaluate_multi_horizon_rollouts(model, X_test_orig, y_test_orig, input_scaler, output_scaler, device, 
                                    horizons=[1, 5, 10, 20, 40], num_rollouts=100):
    """
    Performs multiple short rollouts for various horizon lengths to evaluate
    prediction error accumulation. This is much more representative of MPC usage.
    """
    print("\n--- Evaluating on Test Set (Multi-Horizon Rollouts) ---")
    model.eval()

    # X = [volumes_k (3), state_k (6)] -> state is columns 3 to 9
    num_state_dims = y_test_orig.shape[1]
    state_start_col = 3 # Assumes 3 volume inputs
    state_end_col = state_start_col + num_state_dims
    
    # Store results
    results = {'horizons': [], 'avg_mse': [], 'avg_mae': []}

    for horizon in horizons:
        # Skip if the horizon is longer than the available test data
        if horizon >= len(X_test_orig):
            print(f"Horizon {horizon} is too long for the test set ({len(X_test_orig)} samples). Skipping.")
            continue
            
        horizon_errors_mse = []
        horizon_errors_mae = []
        
        # Determine the possible starting points for a rollout of this length
        max_start_idx = len(X_test_orig) - horizon
        # Randomly select starting points for our rollouts
        start_indices = np.random.choice(max_start_idx, size=min(num_rollouts, max_start_idx), replace=False)
        
        for start_idx in start_indices:
            # --- Perform one short rollout ---
            
            # Get the real starting state from the test set
            current_state = X_test_orig[start_idx, state_start_col:state_end_col]
            
            predicted_trajectory = []
            
            with torch.no_grad():
                for i in range(horizon):
                    # Use the actual control input from the test data
                    control_input = X_test_orig[start_idx + i, :state_start_col]
                    
                    # Combine current state and control input
                    x_and_u = np.concatenate([control_input, current_state])
                    x_and_u_scaled = input_scaler.transform([x_and_u])
                    x_and_u_tensor = torch.tensor(x_and_u_scaled, dtype=torch.float32).to(device)
                    
                    # Predict next state
                    next_state_pred_scaled = model(x_and_u_tensor).cpu().numpy()
                    next_state_pred = output_scaler.inverse_transform(next_state_pred_scaled)[0]
                    
                    predicted_trajectory.append(next_state_pred)
                    current_state = next_state_pred

            # Compare the predicted trajectory to the ground truth
            predicted_trajectory = np.array(predicted_trajectory)
            ground_truth_trajectory = y_test_orig[start_idx : start_idx + horizon]
            
            horizon_errors_mse.append(mean_squared_error(ground_truth_trajectory, predicted_trajectory))
            horizon_errors_mae.append(mean_absolute_error(ground_truth_trajectory, predicted_trajectory))
            
        # Average the errors over all the short rollouts for this horizon
        avg_mse = np.mean(horizon_errors_mse)
        avg_mae = np.mean(horizon_errors_mae)
        
        results['horizons'].append(horizon)
        results['avg_mse'].append(avg_mse)
        results['avg_mae'].append(avg_mae)
        
        print(f"Horizon: {horizon:3d} steps -> Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f}")
    
    # Generate the performance plot
    rollout_plot_path = TrainingConfig.PLOT_OUTPUT_PATH.replace('.png', '_rollout.png')
    plot_rollout_performance(results, rollout_plot_path)

    return results


def train_and_evaluate_architecture(architecture_name, model_class, input_dim, output_dim, 
                                  train_loader, val_loader, sequence_loader, X_test, y_test, 
                                  input_scaler, output_scaler, device, args):
    """Train and evaluate the enhanced architecture with multi-step training."""
    print(f"\n{'='*80}")
    print(f"Training {architecture_name} with Enhanced Multi-Step Learning")
    print(f"{'='*80}")
    
    # Initialize model
    model = model_class(input_dim, output_dim)
    print(f"Model initialized: {architecture_name}")
    
    # Train model with enhanced training
    history = train_model_with_sequences(model, train_loader, sequence_loader, val_loader, 
                                       TrainingConfig.NUM_EPOCHS, TrainingConfig.LEARNING_RATE, 
                                       input_scaler, output_scaler, device)
    
    # Save model
    model_path = TrainingConfig.MODEL_PATH.replace('.pth', f'_{architecture_name.lower().replace(" ", "_")}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # --- Single-step evaluation ---
    print(f"\n--- Evaluating {architecture_name} (Single Step Predictions) ---")
    model.eval()
    X_test_scaled = input_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    predictions = output_scaler.inverse_transform(predictions_scaled)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²):           {r2:.4f}")

    # Generate predictions plot
    plot_path = TrainingConfig.PLOT_OUTPUT_PATH.replace('.png', f'_{architecture_name.lower().replace(" ", "_")}.png')
    plot_predictions(y_test, predictions, plot_path)
    
    results = {
        'architecture': architecture_name,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'model_path': model_path,
        'plot_path': plot_path
    }
    
    # --- Multi-horizon rollout evaluation ---
    if args.rollouts_eval:
        print(f"\n--- Evaluating {architecture_name} (Multi-Horizon Rollouts) ---")
        rollout_results = evaluate_multi_horizon_rollouts(
            model, X_test, y_test, input_scaler, output_scaler, device
        )
        results['rollout_results'] = rollout_results
        
        # Generate rollout plot
        rollout_plot_path = plot_path.replace('.png', '_rollout.png')
        plot_rollout_performance(rollout_results, rollout_plot_path)
        results['rollout_plot_path'] = rollout_plot_path
        
        # Evaluate first-order approximation
        print(f"\n--- Evaluating {architecture_name} (First-Order Approximation Rollouts) ---")
        first_order_results = evaluate_approximation_rollouts(
            model, X_test, y_test, input_scaler, output_scaler, device, 
            approximation_order=1
        )
        results['first_order_results'] = first_order_results
        
        # Evaluate second-order approximation
        print(f"\n--- Evaluating {architecture_name} (Second-Order Approximation Rollouts) ---")
        second_order_results = evaluate_approximation_rollouts(
            model, X_test, y_test, input_scaler, output_scaler, device, 
            approximation_order=2
        )
        results['second_order_results'] = second_order_results
        
        # Generate comparison plot
        comparison_results = {
            'Full Neural Network': rollout_results,
            'First-Order Approximation': first_order_results,
            'Second-Order Approximation': second_order_results
        }
        comparison_plot_path = plot_path.replace('.png', '_approximation_comparison.png')
        plot_approximation_comparison(comparison_results, comparison_plot_path)
        results['comparison_plot_path'] = comparison_plot_path
    
    return results


# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a neural network model for robot state prediction.")
    parser.add_argument('--rollouts-eval', action='store_true', help="Evaluate multi-horizon rollouts after training.")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(TrainingConfig.MODEL_PATH), exist_ok=True)

    # --- Load Data ---
    try:
        X, y = load_and_prepare_data(TrainingConfig.REAL_DATASET_PATH)
        sequences_X, sequences_y = create_sequence_dataset(TrainingConfig.REAL_DATASET_PATH, sequence_length=8)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {TrainingConfig.REAL_DATASET_PATH}")
        print("Please create the file or update the path in the TrainingConfig class.")
        sys.exit()
    except (KeyError, ValueError) as e:
        print(f"Error during data preparation: {e}")
        sys.exit()

    print(f"Total samples loaded: {len(X)}")
    print(f"Shape of input features (X): {X.shape}")
    print(f"Shape of target features (y): {y.shape}")
    print(f"Total sequences loaded: {len(sequences_X)}")
    
    # --- Split Data (using a fixed random_state is crucial for reproducibility) ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=TrainingConfig.VAL_SIZE / (1 - TrainingConfig.TEST_SIZE), random_state=42
    )

    # Split sequence data
    seq_train_val, seq_test = train_test_split(
        list(zip(sequences_X, sequences_y)), test_size=TrainingConfig.TEST_SIZE, random_state=42
    )
    seq_train, seq_val = train_test_split(
        seq_train_val, test_size=TrainingConfig.VAL_SIZE / (1 - TrainingConfig.TEST_SIZE), random_state=42
    )
    
    # Separate sequences back
    seq_X_train, seq_y_train = zip(*seq_train) if seq_train else ([], [])
    seq_X_val, seq_y_val = zip(*seq_val) if seq_val else ([], [])
    
    seq_X_train = np.array(seq_X_train) if seq_X_train else np.array([])
    seq_y_train = np.array(seq_y_train) if seq_y_train else np.array([])
    seq_X_val = np.array(seq_X_val) if seq_X_val else np.array([])
    seq_y_val = np.array(seq_y_val) if seq_y_val else np.array([])

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Training sequences: {len(seq_X_train)}")
    print(f"Validation sequences: {len(seq_X_val)}")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One of the data splits resulted in zero samples. Check your data size and split ratios.")

    # --- Scale Data ---
    print("\nScaling data...")
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)
    
    X_val_scaled = input_scaler.transform(X_val)
    y_val_scaled = output_scaler.transform(y_val)
    print("Data scaled successfully.")
    
    # Scale sequence data
    if len(seq_X_train) > 0:
        # Reshape for scaling: (n_sequences, seq_len, features) -> (n_sequences * seq_len, features)
        seq_X_train_reshaped = seq_X_train.reshape(-1, seq_X_train.shape[-1])
        seq_X_train_scaled = input_scaler.transform(seq_X_train_reshaped)
        seq_X_train_scaled = seq_X_train_scaled.reshape(seq_X_train.shape)
        
        seq_y_train_reshaped = seq_y_train.reshape(-1, seq_y_train.shape[-1])
        seq_y_train_scaled = output_scaler.transform(seq_y_train_reshaped)
        seq_y_train_scaled = seq_y_train_scaled.reshape(seq_y_train.shape)
        
        print("Sequence data scaled successfully.")
    else:
        seq_X_train_scaled = np.array([])
        seq_y_train_scaled = np.array([])
    
    # --- Create DataLoaders ---
    train_dataset = RobotStateDataset(X_train_scaled, y_train_scaled)
    val_dataset = RobotStateDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    # Create sequence dataloaders
    if len(seq_X_train_scaled) > 0:
        sequence_dataset = RobotSequenceDataset(seq_X_train_scaled, seq_y_train)  # Don't scale y for sequences
        sequence_loader = DataLoader(sequence_dataset, batch_size=TrainingConfig.BATCH_SIZE // 2, shuffle=True)
    else:
        # Create empty loader if no sequences
        sequence_loader = DataLoader([], batch_size=1)

    # --- Initialize device and dimensions ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # --- Save scalers ---
    joblib.dump(input_scaler, TrainingConfig.INPUT_SCALER_PATH)
    joblib.dump(output_scaler, TrainingConfig.OUTPUT_SCALER_PATH)
    print(f"Input scaler saved to {TrainingConfig.INPUT_SCALER_PATH}")
    print(f"Output scaler saved to {TrainingConfig.OUTPUT_SCALER_PATH}")

    # --- Train and evaluate the Enhanced Temporal Multi-Scale architecture ---
    try:
        result = train_and_evaluate_architecture(
            "Enhanced Temporal Multi-Scale", StatePredictor, input_dim, output_dim,
            train_loader, val_loader, sequence_loader, X_test, y_test,
            input_scaler, output_scaler, device, args
        )
        
        print(f"\n{'='*80}")
        print("ENHANCED TEMPORAL MULTI-SCALE RESULTS")
        print(f"{'='*80}")
        
        print(f"Single-step Performance:")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  MAE: {result['mae']:.6f}")
        print(f"  R²:  {result['r2']:.4f}")
        
        if 'rollout_results' in result and result['rollout_results']:
            rollout_res = result['rollout_results']
            print(f"\nMulti-step Rollout Performance:")
            for horizon, mse_val in zip(rollout_res['horizons'], rollout_res['avg_mse']):
                print(f"  {horizon:2d}-step MSE: {mse_val:.6f}")
                
        print(f"\nModel saved to: {result['model_path']}")
        print(f"Plots saved to: {os.path.dirname(result['plot_path'])}")
        print("Enhanced training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error training Enhanced Temporal Multi-Scale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)