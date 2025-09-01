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


class BaselineStatePredictor(nn.Module):
    """
    A right-sized neural network for state prediction (Baseline).
    Notes: the model needs to be at least C1 continuous because:
        - uniform continuity assumption 4 (Seel et al., "Neural Network-Based...")
        - differentiability requirement for jacobian computation ("Salzmann et al., "Real-time Neural MPC")
        - eventually C2 for hessian computation (if using second-order approximation)
    
    Derivatives:
        - ReLU: ReLU(x) -> Heaviside(x) (not differentiable at 0)
        - Tanh: tanh(x) -> sech^2(x) -> -2sech^2(x) * tanh(x)
        - SiLU: x * sigmoid(x) -> sigmoid(x) + x * sigmoid'(x) -> sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """
    def __init__(self, input_dim, output_dim):
        super(BaselineStatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connections for better gradient flow."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        residual = x
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return x + residual  # Skip connection


class ResidualStatePredictor(nn.Module):
    """State predictor with residual connections for better training stability."""
    def __init__(self, input_dim, output_dim):
        super(ResidualStatePredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        
        # Residual blocks
        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)
        self.block3 = ResidualBlock(128)
        
        self.output_proj = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.tanh(self.input_proj(x))
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x)
        return self.output_proj(x)


class PhysicsInformedPredictor(nn.Module):
    """Physics-informed architecture separating position and velocity dynamics."""
    def __init__(self, input_dim, output_dim):
        super(PhysicsInformedPredictor, self).__init__()
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        
        # Position dynamics (3 outputs: tip_x, tip_y, tip_z)
        self.position_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        
        # Velocity dynamics (3 outputs: tip_velocity_x, tip_velocity_y, tip_velocity_z)  
        self.velocity_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        features = self.shared(x)
        pos_pred = self.position_net(features)
        vel_pred = self.velocity_net(features)
        return torch.cat([pos_pred, vel_pred], dim=-1)


class WideStatePredictor(nn.Module):
    """Wider network for better expressiveness."""
    def __init__(self, input_dim, output_dim):
        super(WideStatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class MultiScalePredictor(nn.Module):
    """Multi-scale architecture processing inputs at different scales."""
    def __init__(self, input_dim, output_dim):
        super(MultiScalePredictor, self).__init__()
        # Fine-scale features
        self.fine_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Coarse-scale features  
        self.coarse_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh()
        )
        
        # Combine features
        self.combiner = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        fine_features = self.fine_net(x)
        coarse_features = self.coarse_net(x)
        combined = torch.cat([fine_features, coarse_features], dim=-1)
        return self.combiner(combined)


class SiLUStatePredictor(nn.Module):
    """State predictor using SiLU activation for potentially better performance."""
    def __init__(self, input_dim, output_dim):
        super(SiLUStatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)  

# class StatePredictor(nn.Module):
#     """
#     A right-sized neural network for state prediction.
#     Notes: the model needs to be at least C1 continuous because:
#         - uniform continuity assumption 4 (Seel et al., "Neural Network-Based...")
#         - differentiability requirement for jacobian computation ("Salzmann et al., "Real-time Neural MPC")
#         - eventually C2 for hessian computation (if using second-order approximation)
    
#     Derivatives:
#         - ReLU: ReLU(x) -> Heaviside(x) (not differentiable at 0)
#         - Tanh: tanh(x) -> sech^2(x) -> -2sech^2(x) * tanh(x)
#         - SiLU: x * sigmoid(x) -> sigmoid(x) + x * sigmoid'(x) -> sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
#     """
#     def __init__(self, input_dim, output_dim):
#         super(StatePredictor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, output_dim)
#         )
    
#     def forward(self, x):
#         return self.network(x)
    
# This bigger model is more precise but the mpc has slower convergence
"""A bigger, more precise neural network can paradoxically lead to slower MPC convergence because its complexity creates a "jagged" and non-smooth function landscape. The MPC relies on linear approximations (the tangent slope) at each step to find the next best move. On a smooth landscape, these approximations are accurate over a large area, allowing the optimizer to take confident, large steps and converge quickly. On the jagged landscape of the bigger model, the linear approximation is only valid for a tiny, immediate area. This forces the optimizer to take very small, cautious steps and run many more iterations, dramatically slowing down the process. Essentially, for this type of optimization, the smoothness of the model is more important than its absolute precision, and the simpler model provides a much smoother, more navigable landscape for the controller to work with.
"""
# class StatePredictor(nn.Module):
#     """A simple feed-forward neural network for state prediction."""
#     def __init__(self, input_dim, output_dim):
#         super(StatePredictor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
    
#     def forward(self, x):
#         return self.network(x)

class RobotStateDataset(Dataset):
    """Custom PyTorch Dataset."""
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare_data(filepath):
    """
    Loads data from trajectories and creates pairs where X = [u_k, x_k]
    x_k = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    y = [x_k+1].
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

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


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """The main training loop."""
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
                                  train_loader, val_loader, X_test, y_test, 
                                  input_scaler, output_scaler, device, args):
    """Train and evaluate a single architecture."""
    print(f"\n{'='*80}")
    print(f"Training {architecture_name}")
    print(f"{'='*80}")
    
    # Initialize model
    model = model_class(input_dim, output_dim)
    print(f"Model initialized: {architecture_name}")
    
    # Train model
    history = train_model(model, train_loader, val_loader, 
                         TrainingConfig.NUM_EPOCHS, TrainingConfig.LEARNING_RATE, device)
    
    # Save model with architecture-specific name
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
    
    return results


def compare_architectures_performance(all_results, save_path):
    """Generate a comparison plot of all architectures."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    architectures = [r['architecture'] for r in all_results]
    mse_values = [r['mse'] for r in all_results]
    mae_values = [r['mae'] for r in all_results]
    r2_values = [r['r2'] for r in all_results]
    
    # MSE comparison
    bars1 = ax1.bar(architectures, mse_values, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE Comparison Across Architectures')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # MAE comparison
    bars2 = ax2.bar(architectures, mae_values, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('MAE Comparison Across Architectures')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # R² comparison
    bars3 = ax3.bar(architectures, r2_values, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('R-squared (R²)')
    ax3.set_title('R² Comparison Across Architectures')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_values)*0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Rollout performance at horizon 10 (if available)
    rollout_mse_10 = []
    for r in all_results:
        if 'rollout_results' in r and r['rollout_results']:
            rollout_res = r['rollout_results']
            if 10 in rollout_res['horizons']:
                idx = rollout_res['horizons'].index(10)
                rollout_mse_10.append(rollout_res['avg_mse'][idx])
            else:
                rollout_mse_10.append(None)
        else:
            rollout_mse_10.append(None)
    
    # Only plot rollout comparison if we have data
    if any(x is not None for x in rollout_mse_10):
        valid_archs = [arch for arch, val in zip(architectures, rollout_mse_10) if val is not None]
        valid_values = [val for val in rollout_mse_10 if val is not None]
        
        bars4 = ax4.bar(valid_archs, valid_values, color='orange', alpha=0.7)
        ax4.set_ylabel('Rollout MSE (10 steps)')
        ax4.set_title('10-Step Rollout MSE Comparison')
        ax4.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars4, valid_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_values)*0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No rollout data available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('10-Step Rollout MSE Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nArchitecture comparison plot saved to: {save_path}")
    plt.close(fig)


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
    
    # --- Split Data (using a fixed random_state is crucial for reproducibility) ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=TrainingConfig.VAL_SIZE / (1 - TrainingConfig.TEST_SIZE), random_state=42
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    
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
    
    # --- Create DataLoaders ---
    train_dataset = RobotStateDataset(X_train_scaled, y_train_scaled)
    val_dataset = RobotStateDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    # --- Initialize device and dimensions ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # --- Save scalers (shared across all models) ---
    joblib.dump(input_scaler, TrainingConfig.INPUT_SCALER_PATH)
    joblib.dump(output_scaler, TrainingConfig.OUTPUT_SCALER_PATH)
    print(f"Input scaler saved to {TrainingConfig.INPUT_SCALER_PATH}")
    print(f"Output scaler saved to {TrainingConfig.OUTPUT_SCALER_PATH}")

    # --- Define all architectures to train ---
    architectures = [
        ("Baseline", BaselineStatePredictor),
        ("Residual", ResidualStatePredictor),
        ("Physics Informed", PhysicsInformedPredictor),
        ("Wide Network", WideStatePredictor),
        ("Multi Scale", MultiScalePredictor),
        ("SiLU Activation", SiLUStatePredictor)
    ]
    
    # --- Train and evaluate all architectures ---
    all_results = []
    
    for arch_name, arch_class in architectures:
        try:
            result = train_and_evaluate_architecture(
                arch_name, arch_class, input_dim, output_dim,
                train_loader, val_loader, X_test, y_test,
                input_scaler, output_scaler, device, args
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error training {arch_name}: {e}")
            print("Continuing with next architecture...")
            continue
    
    # --- Generate comparison plots and summary ---
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL ARCHITECTURES")
        print(f"{'='*80}")
        
        # Print summary table
        print(f"{'Architecture':<20} {'MSE':<12} {'MAE':<12} {'R²':<10} {'10-step Rollout MSE':<20}")
        print("-" * 80)
        
        for result in all_results:
            rollout_mse_10 = "N/A"
            if 'rollout_results' in result and result['rollout_results']:
                rollout_res = result['rollout_results']
                if 10 in rollout_res['horizons']:
                    idx = rollout_res['horizons'].index(10)
                    rollout_mse_10 = f"{rollout_res['avg_mse'][idx]:.6f}"
            
            print(f"{result['architecture']:<20} {result['mse']:<12.6f} {result['mae']:<12.6f} "
                  f"{result['r2']:<10.4f} {rollout_mse_10:<20}")
        
        # Generate architecture comparison plot
        comparison_plot_path = TrainingConfig.PLOT_OUTPUT_PATH.replace('.png', '_architecture_comparison.png')
        compare_architectures_performance(all_results, comparison_plot_path)
        
        # Find best performing architectures
        best_mse = min(all_results, key=lambda x: x['mse'])
        best_r2 = max(all_results, key=lambda x: x['r2'])
        
        print(f"\nBest MSE: {best_mse['architecture']} (MSE: {best_mse['mse']:.6f})")
        print(f"Best R²:  {best_r2['architecture']} (R²: {best_r2['r2']:.4f})")
        
        if args.rollouts_eval:
            # Find best rollout performance
            rollout_results_valid = [r for r in all_results if 'rollout_results' in r and r['rollout_results']]
            if rollout_results_valid:
                best_rollout = min(rollout_results_valid, 
                                 key=lambda x: x['rollout_results']['avg_mse'][x['rollout_results']['horizons'].index(10)] 
                                 if 10 in x['rollout_results']['horizons'] else float('inf'))
                if 10 in best_rollout['rollout_results']['horizons']:
                    idx = best_rollout['rollout_results']['horizons'].index(10)
                    best_rollout_mse = best_rollout['rollout_results']['avg_mse'][idx]
                    print(f"Best 10-step Rollout: {best_rollout['architecture']} (MSE: {best_rollout_mse:.6f})")
        
        print(f"\nAll models and plots saved to: {os.path.dirname(TrainingConfig.MODEL_PATH)}")
        print("Training and evaluation completed successfully!")
        
    else:
        print("No architectures were successfully trained!")
        sys.exit(1)