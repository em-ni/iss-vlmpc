# src/VLM.py
import requests
import threading
import numpy as np
from queue import Queue, Empty
import sys
import select
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from VLMWebUI import VLMWebUI

def calculate_circle_through_points(p1, p2, p3, num_points=100):
    # Convert input points to numpy arrays
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    
    # Compute two vectors on the plane and the normal
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Create an orthonormal basis in the plane: u along v1 and v perpendicular to u in the plane
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    
    # Project points onto 2D coordinates in the (u,v) plane. Let p1 be the origin.
    a2d = np.array([0, 0])
    b2d = np.array([np.dot(p2 - p1, u), np.dot(p2 - p1, v)])
    c2d = np.array([np.dot(p3 - p1, u), np.dot(p3 - p1, v)])
    
    # Compute circumcenter in 2D for points a2d, b2d, c2d.
    d = 2 * (b2d[0] * c2d[1] - b2d[1] * c2d[0])
    if abs(d) < 1e-6:
        # Points are collinear, so we cannot define a unique circle.
        return np.empty((0, 3))
    center_x = (c2d[1] * (b2d[0]**2 + b2d[1]**2) - b2d[1] * (c2d[0]**2 + c2d[1]**2)) / d
    center_y = (b2d[0] * (c2d[0]**2 + c2d[1]**2) - c2d[0] * (b2d[0]**2 + b2d[1]**2)) / d
    center_2d = np.array([center_x, center_y])
    
    # Convert the 2D center back to the 3D coordinate system.
    center_3d = p1 + center_x * u + center_y * v
    
    # Compute the radius from the 2D center
    radius = np.linalg.norm(b2d - center_2d)
    
    # Compute the angles for p2 and p3 relative to the center in the 2D plane.
    angle_p2 = np.arctan2(b2d[1] - center_y, b2d[0] - center_x)
    angle_p3 = np.arctan2(c2d[1] - center_y, c2d[0] - center_x)
    
    # Ensure we take the minimal angular difference.
    delta = angle_p3 - angle_p2
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi

    # Create an array of angles spanning from p2 to p3.
    angles = np.linspace(angle_p2, angle_p2 + delta, num_points)
    
    # Calculate the arc points in 3D using the (u, v) basis.
    arc_points = []
    for theta in angles:
        point = center_3d + radius * (np.cos(theta) * u + np.sin(theta) * v)
        arc_points.append(point)
    
    return np.array(arc_points)

"""
For Llama.cpp:
brew install llama.cpp
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080

For Gemini:
pip install google-genai python-dotenv
Add G_API_KEY=your-api-key to .env file
"""

class VLM:
    def __init__(self, sim=True,server_url="http://localhost:8080", vlm_dt=1.0, mpc_dt=0.02, backend="llama", model_name="gemini-2.5-pro", web_ui=True):
        """
        Vision Language Model interface for dynamic target assignment.
        
        Args:
            server_url (str): URL of the llama.cpp server (for backend="llama")
            vlm_dt (float): VLM update frequency in seconds
            mpc_dt (float): MPC update frequency in seconds
            backend (str): "llama" for llama.cpp server or "gemini" for Google Gemini
            model_name (str): Model name (for Gemini: "gemini-2.5-pro" or "gemini-2.5-flash")
        """
        self.sim = sim
        self.server_url = server_url
        self.vlm_dt = vlm_dt
        self.mpc_dt = mpc_dt
        self.backend = backend.lower()
        self.model_name = model_name
        self.web_ui = web_ui
        
        # Initialize default attributes first (before any potential exceptions)
        self.session = None
        self.gemini_client = None
        self.user_input_queue = Queue()
        self.input_thread = None
        self.running = False

        if self.web_ui:
            # Initialize UI
            self.ui = VLMWebUI(self.user_input_queue)
        
        # Initialize backend-specific clients
        if self.backend == "llama":
            self.session = requests.Session()
        elif self.backend == "gemini":
            # Look for .env file in current directory and parent directories
            env_paths = [
                '.env',
                '../.env', 
                '../../.env',
                os.path.join(os.path.dirname(__file__), '..', '.env'),
                os.path.join(os.path.dirname(__file__), '..', '..', '.env')
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    break
            
            os.environ['GOOGLE_API_KEY'] = os.getenv('G_API_KEY')
            self.gemini_client = genai.Client()
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'llama' or 'gemini'")
        
        # Predefined targets
        if self.sim:
            self.targets = {
                'right': np.array([0.5, 0.0, -0.5, 0.0, 0.0, 0.0]),
                'left': np.array([-0.5, 0.0, -0.5, 0.0, 0.0, 0.0]),
                'up': np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0]),
                'down': np.array([0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
                'center': np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0])
            }
            self.default_target = self.targets['center']  
        
        # State variables
        self.current_target = None
        self.current_trajectory = None
        self.processing = False
        self.last_response = "VLM initialized. Type commands like 'go right', 'move left', etc."
        self.current_scene_image = None  # Store the latest scene image
        self.waypoints = []  # Store waypoints from VLM

        # Visualization parameters for scene reconstruction
        if self.sim:
            self.xlim = (-1.0, 1.0)
            self.ylim = (-1.0, 1.0)
            self.zlim = (-1.0, 1.0)
        else:
            self.xlim = (0.2, 2.5)
            self.ylim = (-2.0, 1.0)
            self.zlim = (-2.0, 1.0)
        print(f"Workspace limits set to: X{self.xlim}, Y{self.ylim}, Z{self.zlim}")
        self.colors = ['red', 'blue', 'orange', 'purple']
        
        # System prompt for the VLM
        # prompt_filename = "prompt_sim.txt"
        prompt_filename = "prompt_real.txt"
        with open(os.path.join(os.path.dirname(__file__), prompt_filename), "r") as f:
            self.system_prompt = f.read()
            
    def check_server(self):
        """Check if the backend is available."""
        if self.backend == "llama":
            try:
                response = self.session.get(f"{self.server_url}/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        elif self.backend == "gemini":
            try:
                # Only try connection test if we have a valid client
                if self.gemini_client is None:
                    return False
                    
                # Simple test to check if Gemini is accessible
                test_response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=["Test connection"],
                )
                return True
            except Exception as e:
                print(f"Gemini connection test failed: {e}")
                return False
        return False

    def ingest_info_real(self, current_state, robot_base=None, robot_body=None):
        """
        Based on the current state generate the 4 views: XY, XZ, YZ and 3D.
        Combine them in a single plot with 4 subplots
        
        Args:
            current_state: Current state vector containing position
            robot_base: 3D vector for robot base position
            robot_body: 3D vector for robot body center position
            tip_velocity: 3D vector for tip velocity
        """
        try:
            pos = current_state[:3]
            tip_velocity = current_state[3:6] if len(current_state) >= 6 else None
            print(f"Generating scene image at position: {pos}")
            
            # Handle waypoints properly
            waypoints_3d = []
            if self.waypoints and len(self.waypoints) > 0:
                # self.waypoints is a list of 3D tuples [(x1, y1, z1), (x2, y2, z2), ...]
                waypoints_3d = np.array(self.waypoints) 

            # Generate the 4 views
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle("Current State Views", fontsize=16)

            # XY View (Top)
            axs[0, 0].scatter(pos[0], pos[1], c='red', s=100, label='Current Position', 
                            edgecolors='black', linewidth=2, zorder=5)
            
            # Draw robot base as yellow point
            if robot_base is not None:
                axs[0, 0].scatter(robot_base[0], robot_base[1], c='yellow', s=80, 
                                label='Robot Base', edgecolors='black', linewidth=1, zorder=4)
            
            # Draw robot body and circle arc
            if robot_base is not None and robot_body is not None:
                axs[0, 0].scatter(robot_body[0], robot_body[1], c='blue', s=60, 
                                label='Robot Body', edgecolors='black', linewidth=1, zorder=4)
                
                # Draw circle arc through base, body, and tip
                try:
                    # Ensure all parameters are numpy arrays
                    robot_body_arr = np.array(robot_body, dtype=float)
                    pos_arr = np.array(pos, dtype=float)
                    base_arr = np.array([0.0, 0.0, 0.0], dtype=float)
                    
                    circle_points = calculate_circle_through_points(robot_body_arr, pos_arr, base_arr, num_points=50)
                    if circle_points is not None and circle_points.size > 0:
                        axs[0, 0].plot(circle_points[:, 0], circle_points[:, 1], 'b-', 
                                     linewidth=3, alpha=0.7, label='Robot Arc', zorder=3)
                except Exception as e:
                    print(f"Error drawing circle in XY view: {e}")
            
            # Draw tip velocity arrow
            if tip_velocity is not None:
                vel_scale = 0.1  # Scale factor for velocity arrow
                axs[0, 0].arrow(pos[0], pos[1], tip_velocity[0] * vel_scale, tip_velocity[1] * vel_scale,
                              head_width=0.05, head_length=0.03, fc='purple', ec='purple', 
                              linewidth=2, label='Tip Velocity', zorder=6)
            
            if len(waypoints_3d) > 0:
                axs[0, 0].scatter(waypoints_3d[:, 0], waypoints_3d[:, 1], c='green', s=50, 
                                label='Waypoints', alpha=0.7, zorder=4)
            axs[0, 0].set_title("XY View (Top)", fontsize=12)
            axs[0, 0].set_xlabel("X (m)")
            axs[0, 0].set_ylabel("Y (m)")
            axs[0, 0].set_xlim(self.xlim)
            axs[0, 0].set_ylim(self.ylim)
            axs[0, 0].grid(True, alpha=0.3)
            axs[0, 0].legend()
            axs[0, 0].set_aspect('equal')

            # XZ View (Side)
            axs[0, 1].scatter(pos[0], pos[2], c='red', s=100, label='Current Position', 
                            edgecolors='black', linewidth=2, zorder=5)
            
            # Draw robot base as yellow point
            if robot_base is not None:
                axs[0, 1].scatter(robot_base[0], robot_base[2], c='yellow', s=80, 
                                label='Robot Base', edgecolors='black', linewidth=1, zorder=4)
            
            # Draw robot body and circle arc
            if robot_base is not None and robot_body is not None:
                axs[0, 1].scatter(robot_body[0], robot_body[2], c='orange', s=60, 
                                label='Robot Body', edgecolors='black', linewidth=1, zorder=4)
                
                # Draw circle arc through base, body, and tip
                try:
                    # Ensure all parameters are numpy arrays
                    robot_body_arr = np.array(robot_body, dtype=float)
                    pos_arr = np.array(pos, dtype=float)
                    base_arr = np.array([0.0, 0.0, 0.0], dtype=float)
                    
                    circle_points = calculate_circle_through_points(robot_body_arr, pos_arr, base_arr, num_points=50)
                    if circle_points is not None and circle_points.size > 0:
                        axs[0, 1].plot(circle_points[:, 0], circle_points[:, 2], 'b-', 
                                     linewidth=3, alpha=0.7, label='Robot Arc', zorder=3)
                except Exception as e:
                    print(f"Error drawing circle in XZ view: {e}")
            
            # Draw tip velocity arrow
            if tip_velocity is not None:
                vel_scale = 0.1  # Scale factor for velocity arrow
                axs[0, 1].arrow(pos[0], pos[2], tip_velocity[0] * vel_scale, tip_velocity[2] * vel_scale,
                              head_width=0.05, head_length=0.03, fc='purple', ec='purple', 
                              linewidth=2, label='Tip Velocity', zorder=6)
            
            if len(waypoints_3d) > 0:
                axs[0, 1].scatter(waypoints_3d[:, 0], waypoints_3d[:, 2], c='green', s=50, 
                                label='Waypoints', alpha=0.7, zorder=4)
            axs[0, 1].set_title("XZ View (Side)", fontsize=12)
            axs[0, 1].set_xlabel("X (m)")
            axs[0, 1].set_ylabel("Z (m)")
            axs[0, 1].set_xlim(self.xlim)
            axs[0, 1].set_ylim(self.zlim)
            axs[0, 1].grid(True, alpha=0.3)
            axs[0, 1].legend()
            axs[0, 1].set_aspect('equal')

            # YZ View (Front)
            axs[1, 0].scatter(pos[1], pos[2], c='red', s=100, label='Current Position', 
                            edgecolors='black', linewidth=2, zorder=5)
            
            # Draw robot base as yellow point
            if robot_base is not None:
                axs[1, 0].scatter(robot_base[1], robot_base[2], c='yellow', s=80, 
                                label='Robot Base', edgecolors='black', linewidth=1, zorder=4)
            
            # Draw robot body and circle arc
            if robot_base is not None and robot_body is not None:
                axs[1, 0].scatter(robot_body[1], robot_body[2], c='orange', s=60, 
                                label='Robot Body', edgecolors='black', linewidth=1, zorder=4)
                
                # Draw circle arc through base, body, and tip
                try:
                    # Ensure all parameters are numpy arrays
                    robot_body_arr = np.array(robot_body, dtype=float)
                    pos_arr = np.array(pos, dtype=float)
                    base_arr = np.array([0.0, 0.0, 0.0], dtype=float)
                    
                    circle_points = calculate_circle_through_points(robot_body_arr, pos_arr, base_arr, num_points=50)
                    if circle_points is not None and circle_points.size > 0:
                        axs[1, 0].plot(circle_points[:, 1], circle_points[:, 2], 'b-', 
                                     linewidth=3, alpha=0.7, label='Robot Arc', zorder=3)
                except Exception as e:
                    print(f"Error drawing circle in YZ view: {e}")
            
            # Draw tip velocity arrow
            if tip_velocity is not None:
                vel_scale = 0.1  # Scale factor for velocity arrow
                axs[1, 0].arrow(pos[1], pos[2], tip_velocity[1] * vel_scale, tip_velocity[2] * vel_scale,
                              head_width=0.05, head_length=0.03, fc='purple', ec='purple', 
                              linewidth=2, label='Tip Velocity', zorder=6)
            
            if len(waypoints_3d) > 0:
                axs[1, 0].scatter(waypoints_3d[:, 1], waypoints_3d[:, 2], c='green', s=50, 
                                label='Waypoints', alpha=0.7, zorder=4)
            axs[1, 0].set_title("YZ View (Front)", fontsize=12)
            axs[1, 0].set_xlabel("Y (m)")
            axs[1, 0].set_ylabel("Z (m)")
            axs[1, 0].set_xlim(self.ylim)
            axs[1, 0].set_ylim(self.zlim)
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].legend()
            axs[1, 0].set_aspect('equal')

            # 3D View
            ax_3d = fig.add_subplot(224, projection='3d')
            ax_3d.scatter(pos[0], pos[1], pos[2], c='red', s=100, label='Current Position', 
                        edgecolors='black', linewidth=2, zorder=5)
            
            # Draw robot base as yellow point
            if robot_base is not None:
                ax_3d.scatter(robot_base[0], robot_base[1], robot_base[2], c='yellow', s=80, 
                            label='Robot Base', edgecolors='black', linewidth=1, zorder=4)
            
            # Draw robot body and circle arc
            if robot_base is not None and robot_body is not None:
                ax_3d.scatter(robot_body[0], robot_body[1], robot_body[2], c='orange', s=60, 
                            label='Robot Body', edgecolors='black', linewidth=1, zorder=4)
                
                # Draw circle arc through base, body, and tip
                try:
                    # Ensure all parameters are numpy arrays
                    robot_body_arr = np.array(robot_body, dtype=float)
                    pos_arr = np.array(pos, dtype=float)
                    base_arr = np.array([0.0, 0.0, 0.0], dtype=float)
                    
                    circle_points = calculate_circle_through_points(robot_body_arr, pos_arr, base_arr, num_points=50)
                    if circle_points is not None and circle_points.size > 0:
                        ax_3d.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'b-', 
                                 linewidth=3, alpha=0.7, label='Robot Arc', zorder=3)
                except Exception as e:
                    print(f"Error drawing circle in 3D view: {e}")
            
            # Draw tip velocity arrow
            if tip_velocity is not None:
                vel_scale = 0.1  # Scale factor for velocity arrow
                ax_3d.quiver(pos[0], pos[1], pos[2], 
                           tip_velocity[0] * vel_scale, tip_velocity[1] * vel_scale, tip_velocity[2] * vel_scale,
                           color='purple', arrow_length_ratio=0.15, linewidth=2, 
                           label='Tip Velocity', alpha=0.8)
            
            # Add text label for current position
            ax_3d.text(pos[0], pos[1], pos[2], f'  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})', 
                      fontsize=8, color='red', weight='bold')
            
            if len(waypoints_3d) > 0:
                ax_3d.scatter(waypoints_3d[:, 0], waypoints_3d[:, 1], waypoints_3d[:, 2], 
                            c='green', s=50, label='Waypoints', alpha=0.7, zorder=4)
                
                # Add text labels for waypoints
                for i, wp in enumerate(waypoints_3d):
                    ax_3d.text(wp[0], wp[1], wp[2], f'  WP{i+1}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})', 
                              fontsize=8, color='green')
                
                # Draw trajectory line if waypoints exist
                trajectory_points = np.vstack([pos.reshape(1, -1), waypoints_3d])
                ax_3d.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                        'g--', alpha=0.5, linewidth=2, label='Planned Path', zorder=3)
            
            ax_3d.set_title("3D View", fontsize=12)
            ax_3d.set_xlabel("X (m)")
            ax_3d.set_ylabel("Y (m)")
            ax_3d.set_zlabel("Z (m)")
            ax_3d.set_xlim(self.xlim)
            ax_3d.set_ylim(self.ylim)
            ax_3d.set_zlim(self.zlim)
            ax_3d.legend()

            plt.tight_layout()
            
            # Convert to base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)  # Clean up to prevent memory leaks
            
            # Store for later use
            self.current_scene_image = image_base64
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating real scene image: {e}")
            return None
        
    def ingest_info_sim(self, sim_data, current_target=None, tip_history=None, target_history=None):
        """
        Create visual representation of the current scene for VLM processing.
        
        Args:
            sim_data: Simulation object
            current_target (np.array): Current target position [x, y, z]
            tip_history (list): History of tip positions
            target_history (list): History of target positions
            
        Returns:
            str: Base64 encoded image of the scene
        """
        try:
            # Get rods data from simulation
            rods = sim_data.get_rods()
            if not rods:
                return None
                
            # Create a simpler, single view figure for better VLM understanding
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title("Robot Workspace Scene XY View")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            
            # Plot robot tip position in black
            current_tip = rods[-1].position_collection[:,-1]
            ax.plot(current_tip[0], current_tip[1], 'ko', markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, label='ROBOT TIP', zorder=5)
            
            # Plot current target position in green
            if current_target is not None:
                ax.plot(current_target[0], current_target[1], 'go', markersize=10, 
                       markeredgecolor='black', markeredgewidth=2, label='CURRENT TARGET', zorder=5)
            else:
                current_target = np.array([0.0, 0.0])
            
            # Extract and plot targets directly from simulation
            if hasattr(sim_data, 'get_targets'):
                sim_targets = sim_data.get_targets()
                for target in sim_targets:
                    # Get target position
                    if hasattr(target, 'position_collection'):
                        if target.position_collection.ndim == 1:
                            target_pos = target.position_collection
                        else:
                            target_pos = target.position_collection[:, 0]
                    else:
                        continue  # Skip if no position data
                    
                    # Get target color (use the stored target_color attribute)
                    target_color = getattr(target, 'target_color', 'gray')
                    target_id = getattr(target, 'target_id', 'unknown')
                    
                    # Plot the target with its correct color
                    ax.plot(target_pos[0], target_pos[1], 'o', color=target_color,
                           markersize=8, markeredgecolor='black', markeredgewidth=1,
                           label=f'Target {target_id}: {target_color.upper()}', zorder=4)

            # Extract and plot obstacles directly from simulation
            if hasattr(sim_data, 'get_obstacles'):
                sim_obstacles = sim_data.get_obstacles()
                for obstacle in sim_obstacles:
                    # Get obstacle position
                    if hasattr(obstacle, 'position_collection'):
                        if obstacle.position_collection.ndim == 1:
                            obstacle_pos = obstacle.position_collection
                        else:
                            obstacle_pos = obstacle.position_collection[:, 0]
                    else:
                        continue  # Skip if no position data
                    
                    # Get obstacle properties
                    obstacle_color = getattr(obstacle, 'obstacle_color', 'gray')
                    obstacle_id = getattr(obstacle, 'obstacle_id', 'unknown')
                    obstacle_radius = getattr(obstacle, 'base_radius', 0.05)
                    
                    # Plot the obstacle as a circle (cross-section of cylinder)
                    circle = patches.Circle((obstacle_pos[0], obstacle_pos[1]), obstacle_radius,
                                          color=obstacle_color, alpha=0.7, zorder=3,
                                          edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                    # ax.text(obstacle_pos[0], obstacle_pos[1] + obstacle_radius + 0.05, 
                    #        f'Obstacle {obstacle_id}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            
            # Convert to base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')  # Lower DPI for faster processing
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)  # Clean up
            
            # Store for later use
            self.current_scene_image = image_base64
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating scene image: {e}")
            return None

    def query_vlm(self, user_input, scene_image=None):
        """Query the VLM with user input and optional scene image to get target direction."""
        if self.processing:
            return None
            
        self.processing = True
        try:
            if self.backend == "llama":
                return self._query_llama(user_input, scene_image)
            elif self.backend == "gemini":
                return self._query_gemini(user_input, scene_image)
        except Exception as e:
            error_msg = f"VLM Error: {str(e)}"
            print(f"VLM exception: {error_msg}")
            self.last_response = error_msg
            return None
        finally:
            self.processing = False

    def _query_llama(self, user_input, scene_image=None):
        """Query Llama.cpp server."""
        # Prepare the message content
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Create user message with text and optional image
        user_message = {"role": "user", "content": []}
        
        # Add text content
        user_message["content"].append({
            "type": "text",
            "text": user_input
        })
        
        # Add image if provided
        if scene_image or self.current_scene_image:
            image_data = scene_image or self.current_scene_image
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })
        
        messages.append(user_message)
        
        payload = {
            "model": "gpt-4-vision-preview",  # This is just for compatibility
            "max_tokens": 10,  # Reduced to force shorter responses
            "temperature": 0.1,  # Lower temperature for more consistent responses
            "messages": messages
        }
        
        print(f"Sending Llama VLM query: '{user_input}' with {'image' if (scene_image or self.current_scene_image) else 'text only'}")
        
        response = self.session.post(
            f"{self.server_url}/v1/chat/completions", 
            json=payload, 
            timeout=15  # Slightly longer timeout for image processing
        )
        
        print(f"Llama server response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            raw_result = data["choices"][0]["message"]["content"].strip()
            
            print(f"Llama raw response: '{raw_result}'")
            
            # Clean up the response - remove quotes, periods, and extra characters
            cleaned_result = raw_result.lower()
            cleaned_result = cleaned_result.replace('"', '').replace("'", "")
            cleaned_result = cleaned_result.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
            cleaned_result = cleaned_result.strip()
            
            # Extract only the first word if response is too long
            first_word = cleaned_result.split()[0] if cleaned_result.split() else cleaned_result
            
            print(f"Llama cleaned response: '{first_word}'")
            
            image_info = " (with image)" if (scene_image or self.current_scene_image) else " (text only)"
            self.last_response = f"Llama response: '{first_word}' from input: '{user_input}'{image_info}"
            return first_word
        else:
            error_msg = f"Llama server error {response.status_code}: {response.text}"
            print(f"Llama server error: {error_msg}")
            self.last_response = error_msg
            return None

    def _query_gemini(self, user_input, scene_image=None):
        """Query Google Gemini. Return target coordinates as a string: x,y"""
        print(f"Sending Gemini query: '{user_input}' with {'image' if (scene_image or self.current_scene_image) else 'text only'}")
        
        # Prepare contents for Gemini
        contents = []
        
        # Add system instruction through the user prompt for now
        # (Gemini has system instructions but this approach is simpler)
        full_prompt = f"{self.system_prompt}\n\nUser command: {user_input}"
        contents.append(full_prompt)
        
        # Add image if provided and not in text-only mode
        if (scene_image or self.current_scene_image):
            image_data = scene_image or self.current_scene_image
            
            # Convert base64 to bytes for Gemini
            try:
                image_bytes = base64.b64decode(image_data)
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                )
                contents.append(image_part)
            except Exception as e:
                print(f"Error processing image for Gemini: {e}")
                # Continue without image
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            self.last_response = response.text
            raw_result = response.text
            print(f"Gemini raw response: {raw_result}")
            return raw_result.strip()
            
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print(f"Gemini error: {error_msg}")
            self.last_response = error_msg
            return "0.0,0.0"

    def generate_trajectory(self, current_state, target_state, transition_time=5.0):
        """
        Generate a smooth trajectory from current state to target state.
        
        Args:
            current_state (np.array): Current 6D state [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
            target_state (np.array): Target 6D state
            transition_time (float): Time to reach target in seconds
            
        Returns:
            np.array: Trajectory array of shape (n_steps, 6)
        """
        n_steps = int(transition_time / self.mpc_dt)
        if n_steps < 1:
            n_steps = 1
            
        # Generate smooth trajectory (linear interpolation for now)
        trajectory = np.zeros((n_steps, 6))
        for i in range(n_steps):
            alpha = i / (n_steps - 1) if n_steps > 1 else 1.0
            # Smooth interpolation using sigmoid-like function
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
            trajectory[i] = current_state + smooth_alpha * (target_state - current_state)
            
        return trajectory

    def generate_trajectory_from_waypoints(self, current_state, waypoints, transition_time=5.0, wp_hold_steps=20):
        """
        Generate a trajectory from current state to a series of waypoints.
        
        Args:
            current_state (np.array): Current 6D state [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
            waypoints (list of float): For sim=True: [x1,y1,x2,y2,...,xn,yn] 
                                      For sim=False: [x1,y1,z1,x2,y2,z2,...,xn,yn,zn]
            transition_time (float): Time to reach target in seconds
            
        Returns:
            np.array: Trajectory array of shape (n_steps, 6)
        """
        
        # Check if waypoints is empty or invalid
        coords_per_waypoint = 2 if self.sim else 3
        if not waypoints or len(waypoints) < coords_per_waypoint:
            print("No valid waypoints provided")
            return None

        # Create target states from waypoints
        target_states = []
        for i in range(0, len(waypoints), coords_per_waypoint):
            if i + coords_per_waypoint - 1 < len(waypoints):  # Ensure we have all coordinates
                if self.sim:
                    # 2D waypoints: use default z=-0.5
                    target_states.append(np.array([waypoints[i], waypoints[i+1], -0.5, 0.0, 0.0, 0.0]))
                else:
                    # 3D waypoints: use provided z coordinate
                    target_states.append(np.array([waypoints[i], waypoints[i+1], waypoints[i+2], 0.0, 0.0, 0.0]))

        if not target_states:
            print("No valid target states created from waypoints")
            return None

        # Generate trajectory through each waypoint
        full_trajectory = []
        for target_state in target_states:
            trajectory = self.generate_trajectory(current_state, target_state, transition_time)
            # Append hold steps at the end of each segment
            if wp_hold_steps > 0:
                hold_state = target_state.copy()
                hold_state[3:] = 0.0
                hold_trajectory = np.tile(hold_state, (wp_hold_steps, 1))
                trajectory = np.concatenate((trajectory, hold_trajectory))
            full_trajectory.append(trajectory)
            current_state = target_state

        if not full_trajectory:
            print("No trajectory segments generated")
            return None
            
        return np.concatenate(full_trajectory)

    def start_input_thread(self):
        self.running = True
        if self.web_ui:
            print("Starting VLM UI...")
            self.ui.start_ui()
            print("VLM UI started! Use the GUI window to send commands.")
        else:
            self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
            self.input_thread.start()
            print("VLM input thread started. Type commands during simulation!")

    def _input_worker(self):
        """Worker thread to handle user input."""
        while self.running:
            try:
                # Non-blocking input check
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = input().strip()
                    if user_input:
                        self.user_input_queue.put(user_input)
                        print(f"Command queued: '{user_input}'")
            except Exception as e:
                # Handle input errors gracefully
                pass

    def process_user_input(self, current_state, scene_image=None):
        """
        Process any pending user input and return new trajectory if needed.
        
        Args:
            current_state (np.array): Current robot state
            scene_image (str): Base64 encoded scene image (optional)
            
        Returns:
            tuple: (new_trajectory, target_name) or (None, None) if no new command
        """
        try:
            # Check for new user input
            user_input = self.user_input_queue.get_nowait()
            print(f"Processing command: '{user_input}'")

            if self.web_ui: self.ui.add_status_update(f"Processing: {user_input}")

            # First try with scene image if available
            vlm_response = None
            if scene_image:
                vlm_response = self.query_vlm(user_input, scene_image)
            
            if vlm_response is None:
                print("VLM query failed")
                if self.web_ui: self.ui.add_response("❌ VLM query failed")
                return None, None
                
            # Add VLM response to UI
            if self.web_ui: self.ui.add_response(f"Target: {vlm_response}")
            # Handle stop command
            if vlm_response == "stop":
                print("Stop command received")
                if self.web_ui: self.ui.add_status_update("Robot stopped")
                # Create a trajectory that stays at current position
                stop_target = current_state.copy()
                stop_target[3:] = 0.0  # Zero velocities
                trajectory = self.generate_trajectory(current_state, stop_target, 0.5)
                return trajectory, "stop"
            
            # Handle movement commands
            try:
                coords = vlm_response.split(',')
                
                if self.sim:
                    # For simulation: 2D coordinates (x1,y1,x2,y2,...,xn,yn)
                    if len(coords) % 2 != 0:
                        raise ValueError("Odd number of coordinates for 2D waypoints")
                    
                    # Fill waypoint list as flat list for generate_trajectory_from_waypoints
                    waypoints_flat = []
                    waypoints_tuples = []  # For validation and storage
                    
                    for i in range(0, len(coords), 2):
                        if i + 1 >= len(coords):
                            raise ValueError("Missing y coordinate")
                            
                        x = float(coords[i].strip())
                        y = float(coords[i + 1].strip())
                        
                        # Validate bounds
                        if not (self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]):
                            raise ValueError(f"Waypoint ({x}, {y}) out of bounds")
                        
                        waypoints_flat.extend([x, y])
                        waypoints_tuples.append((x, y))
                else:
                    # For real robot: 3D coordinates (x1,y1,z1,x2,y2,z2,...,xn,yn,zn)
                    if len(coords) % 3 != 0:
                        raise ValueError("Number of coordinates must be multiple of 3 for 3D waypoints")
                    
                    # Fill waypoint list as flat list for generate_trajectory_from_waypoints
                    waypoints_flat = []
                    waypoints_tuples = []  # For validation and storage
                    
                    for i in range(0, len(coords), 3):
                        if i + 2 >= len(coords):
                            raise ValueError("Missing y or z coordinate")
                            
                        x = float(coords[i].strip())
                        y = float(coords[i + 1].strip())
                        z = float(coords[i + 2].strip())
                        
                        # Validate bounds
                        if not (self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1] and self.zlim[0] <= z <= self.zlim[1]):
                            raise ValueError(f"Waypoint ({x}, {y}, {z}) out of bounds")
                        
                        waypoints_flat.extend([x, y, z])
                        waypoints_tuples.append((x, y, z))
                
                # Generate trajectory from waypoints (expects flat list)
                trajectory = self.generate_trajectory_from_waypoints(current_state, waypoints_flat)

                if trajectory is None:
                    print(f"Failed to generate trajectory from waypoints: '{vlm_response}'")
                    if self.web_ui: self.ui.add_response("❌ Failed to generate trajectory")
                    return None, None

                # Store waypoints as tuples for easier access
                self.waypoints = waypoints_tuples
                # Get the last waypoint coordinates for current target
                last_waypoint = self.waypoints[-1]
                if self.sim:
                    self.current_target = f"{last_waypoint[0]},{last_waypoint[1]}"
                else:
                    self.current_target = f"{last_waypoint[0]},{last_waypoint[1]},{last_waypoint[2]}"
                self.current_trajectory = trajectory
                print(f"Generated trajectory with waypoints: {self.waypoints}")
                
                return trajectory, vlm_response

            except ValueError:
                print(f"Could not parse coordinates from response: '{vlm_response}', defaulting to center")
                if self.web_ui: self.ui.add_response("❌ Could not parse coordinates")
                return None, None
                
        except Empty:
            # No new input
            return None, None
        except Exception as e:
            print(f"Error processing input: {e}")
            return None, None

    def save_scene_image(self, filename=None):
        """Save the current scene image to disk for debugging."""
        if self.current_scene_image is None:
            print("No scene image available to save")
            return False
            
        try:
            import os
            if filename is None:
                filename = f"vlm_scene_{int(time.time())}.png"
                
            # Ensure results directory exists
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filename)
            
            # Decode and save image
            image_data = base64.b64decode(self.current_scene_image)
            with open(filepath, 'wb') as f:
                f.write(image_data)
                
            print(f"Scene image saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving scene image: {e}")
            return False

    def get_status(self):
        """Get current VLM status information."""
        return {
            'processing': self.processing,
            'current_target': self.current_target,
            'last_response': self.last_response,
            'backend': self.backend,
            'model_name': self.model_name if self.backend == "gemini" else "llama.cpp",
            'server_connected': self.check_server(),
            'queue_size': self.user_input_queue.qsize()
        }

    def stop(self):
        """Stop the VLM and cleanup."""
        self.running = False
        
        if self.web_ui:
            # Stop UI
            if hasattr(self, 'ui') and self.ui:
                self.ui.stop()
            
        # Stop input thread (legacy)
        if hasattr(self, 'input_thread') and self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
            
        print("VLM stopped")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop()
        except Exception:
            # Ignore errors during cleanup
            pass


    