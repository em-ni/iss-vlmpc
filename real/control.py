import socket
import struct
import sys
import os
import threading
import time
import numpy as np
from queue import Queue, Empty

# Local imports
import src.config as config
from src.tracker import Tracker
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add MPC project path
mpc_project_path = os.path.join(project_root, 'PSS-VLMPC', 'generic-neural-mpc')
if mpc_project_path not in sys.path:
    sys.path.append(mpc_project_path)
from mpc_casadi_real import MPCController

# Add VLM import path
vlm_path = os.path.join(project_root, 'PSS-VLMPC', 'sim', 'src')
if vlm_path not in sys.path:
    sys.path.append(vlm_path)
from VLM import VLM

# Simulation settings
FINAL_TIME = 10.0
CONTROL_MODE = "vlm"  # set point regulation, "tt" for trajectory tracking, "vlm" for VLM control
APPROXIMATION_ORDER = 2

# Real robot parameters
simulation_params = {
    'mpc_dt': 0.1,     # 10Hz
    'vlm_dt': 10.0      # 1Hz
}

class ThreadedRobotController:
    def __init__(self):
        # Control flags
        self.quit = False
        
        # Thread-safe data sharing
        self.state_lock = threading.Lock()
        self.trajectory_lock = threading.Lock()
        self.control_lock = threading.Lock()
        
        # Shared state variables
        self.current_state = None
        self.current_control = None
        self.vlm_trajectory = None
        self.vlm_trajectory_index = 0
        self.vlm_target_name = None
        
        # Components
        self.tracker = None
        self.mpc = None
        self.vlm = None
        
        # Threads
        self.tracker_thread = None
        self.vlm_thread = None
        self.mpc_thread = None
        
        # History for plotting
        self.history_x = []
        self.history_u = []
        self.history_x_target = []
        self.history_mpc_times = []
        
    def get_current_state_safe(self):
        """Thread-safe getter for current state"""
        with self.state_lock:
            return self.current_state.copy() if self.current_state is not None else None
    
    def update_vlm_trajectory_safe(self, new_trajectory, target_name):
        """Thread-safe setter for VLM trajectory"""
        with self.trajectory_lock:
            self.vlm_trajectory = new_trajectory
            self.vlm_trajectory_index = 0
            self.vlm_target_name = target_name
            
    def get_vlm_target_safe(self):
        """Thread-safe getter for current VLM target"""
        with self.trajectory_lock:
            if self.vlm_trajectory is not None:
                if self.vlm_trajectory_index < len(self.vlm_trajectory):
                    target = self.vlm_trajectory[self.vlm_trajectory_index].copy()
                    self.vlm_trajectory_index += 1
                    return target
                else:
                    return self.vlm_trajectory[-1].copy()
            return None
    
    def update_control_safe(self, control):
        """Thread-safe setter for control command"""
        with self.control_lock:
            self.current_control = control.copy()
            
    def get_control_safe(self):
        """Thread-safe getter for control command"""
        with self.control_lock:
            return self.current_control.copy() if self.current_control is not None else None

    def vlm_worker_thread(self):
        """Dedicated thread for VLM processing at 1Hz"""
        print("VLM worker thread started")
        target_dt = simulation_params['vlm_dt'] 
        saved_first_image = False
        while not self.quit:
            cycle_start_time = time.time()
            
            try:
                # Get current state (thread-safe)
                current_state = self.get_current_state_safe()
                if current_state is None:
                    print("VLM: Waiting for initial state...")
                    time.sleep(0.1)
                    continue
                
                # Generate scene image (you'll need to implement this based on your setup)
                scene_image = self.vlm.ingest_info_real(current_state)

                # Save first scene image for debugging
                if not saved_first_image and scene_image is not None:
                    self.vlm.save_scene_image(filename='initial_vlm_view.png')
                    print("Initial scene image saved as 'initial_vlm_view.png'")
                    saved_first_image = True

                # Process any pending user input
                new_trajectory, target_name = self.vlm.process_user_input(current_state, scene_image)
                
                if new_trajectory is not None:
                    self.update_vlm_trajectory_safe(new_trajectory, target_name)
                    print(f"VLM: New trajectory activated to reach: {target_name}")
                
            except Exception as e:
                print(f"VLM thread error: {e}")
                # Continue running even if VLM fails
            
            # Time-compensated sleep for 1Hz
            elapsed = time.time() - cycle_start_time
            remaining_time = target_dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
            else:
                print(f"VLM: Processing took {elapsed:.3f}s, missed target of {target_dt}s")

    def mpc_worker_thread(self):
        """Dedicated thread for MPC processing
        MPC computation takes ~70ms
        """
        print("MPC worker thread started")
        target_dt = simulation_params['mpc_dt']  
        
        # Default target
        x_target = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        while not self.quit:
            cycle_start_time = time.time()
            
            try:
                # Get current state (thread-safe)
                current_state = self.get_current_state_safe()
                if current_state is None:
                    print("MPC: Waiting for initial state...")
                    time.sleep(0.1)
                    continue
                
                # Update target from VLM if available
                vlm_target = self.get_vlm_target_safe()
                if vlm_target is not None:
                    x_target = vlm_target
                
                # Compute MPC control
                start_mpc_time = time.time()
                u_mpc = self.mpc.step(x_target, current_state)
                end_mpc_time = time.time()
                
                mpc_computation_time = end_mpc_time - start_mpc_time
                self.history_mpc_times.append(mpc_computation_time)
                
                if u_mpc is None:
                    print("MPC: Control computation failed")
                    continue
                
                # Update control command (thread-safe)
                self.update_control_safe(np.array(u_mpc))
                
                # Store history (you might want to make this thread-safe too)
                self.history_x.append(current_state.copy())
                self.history_u.append(u_mpc.copy())
                self.history_x_target.append(x_target.copy())
                
            except Exception as e:
                print(f"MPC thread error: {e}")
                # Continue running even if MPC fails
            
            # Time-compensated sleep
            elapsed = time.time() - cycle_start_time
            remaining_time = target_dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
            else:
                print(f"Warning: MPC Processing took {elapsed:.3f}s, missed target of {target_dt}s")

    def update_state_from_tracker(self):
        """Update shared state from tracker data"""
        try:
            state_with_timestamp = self.tracker.get_current_state()
            if state_with_timestamp is not None:
                state = state_with_timestamp[:-1]  # Exclude timestamp
                with self.state_lock:
                    self.current_state = state
                    # print(f"Updated state: {self.current_state}")
        except Exception as e:
            print(f"Error updating state from tracker: {e}")

    def send_robot_command(self):
        """Send control command to robot at high frequency"""
        control = self.get_control_safe()
        if control is not None:
            # TODO: Implement actual robot command sending
            # send_robot_command_implementation(control)
            pass

def send_quit_signal():
    """Send a quit signal to the tracker via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = struct.pack('?', True)
    sock.sendto(data, (config.UDP_IP, config.UDP_QUIT_TRACK_PORT))
    sock.close()
    print("Sent quit signal to tracker.")

def main():
    global CONTROL_MODE
    
    print("Initializing Real Robot MPC Control...")
    
    # Create controller instance
    controller = ThreadedRobotController()
    
    try:
        # Initialize Tracker
        print("Initializing Tracker...")
        experiment_name = config.experiment_name
        save_dir = config.save_dir
        csv_path = config.csv_path
        controller.tracker = Tracker(experiment_name, save_dir, csv_path, realtime=True)

        # Start tracker in a thread
        controller.tracker_thread = threading.Thread(target=controller.tracker.run_realtime_tracking, args=(True,))
        controller.tracker_thread.start()
        time.sleep(2)
        print("Tracker thread started.")
        
        # Initialize VLM if needed
        if CONTROL_MODE == "vlm":
            print("Initializing VLM...")
            controller.vlm = VLM(
                sim=False,
                vlm_dt=simulation_params['vlm_dt'],
                mpc_dt=simulation_params['mpc_dt'],
                backend="gemini",
                model_name="gemini-2.5-pro",
                web_ui=True
            )

            if not controller.vlm.check_server():
                print("Warning: VLM server not running! Switching to set point regulation mode.")
                CONTROL_MODE = 'spr'
            else:
                print("VLM server connected successfully!")
                controller.vlm.start_input_thread()
                
                # Start VLM worker thread
                controller.vlm_thread = threading.Thread(target=controller.vlm_worker_thread)
                controller.vlm_thread.start()
                print("VLM worker thread started.")

        # Initialize MPC controller
        print("Initializing MPC Controller...")
        controller.mpc = MPCController(nn_approximation_order=APPROXIMATION_ORDER)

        # Start MPC worker thread
        controller.mpc_thread = threading.Thread(target=controller.mpc_worker_thread)
        controller.mpc_thread.start()
        print("MPC worker thread started.")
        
        # Main loop for high-frequency operations
        print("Starting main control loop...")
        main_loop_dt = 0.01  # 100Hz for robot command sending
        
        while True:
            loop_start_time = time.time()
            
            # Update state from tracker
            controller.update_state_from_tracker()
            
            # # Send robot command at high frequency
            # controller.send_robot_command()
            
            # Time-compensated sleep for main loop
            elapsed = time.time() - loop_start_time
            remaining_time = main_loop_dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)

    except KeyboardInterrupt:
        print("\nControl interrupted by user")
        controller.quit = True
        
        # Stop all threads
        print("Stopping VLM...")
        if controller.vlm:
            controller.vlm.stop()
        
        print("Sending quit signal to tracker...")
        send_quit_signal()
        
        # Join all threads
        if controller.vlm_thread and controller.vlm_thread.is_alive():
            controller.vlm_thread.join(timeout=2.0)
            print("VLM thread joined.")
            
        if controller.mpc_thread and controller.mpc_thread.is_alive():
            controller.mpc_thread.join(timeout=2.0)
            print("MPC thread joined.")
            
        if controller.tracker_thread and controller.tracker_thread.is_alive():
            controller.tracker_thread.join(timeout=2.0)
            print("Tracker thread joined.")
            
    except Exception as e:
        print(f"Error during control: {e}")
        import traceback
        traceback.print_exc()
        controller.quit = True
    
    # Print results
    if controller.history_mpc_times:
        avg_mpc_time = np.mean(controller.history_mpc_times)
        print(f"\nAverage MPC computation time: {avg_mpc_time:.4f}s")
    
    # Plot results
    if controller.history_x and controller.history_u:
        controller.mpc.history_x = controller.history_x
        controller.mpc.history_u = controller.history_u
        controller.mpc.plot_results(history_x_target=controller.history_x_target)
        print("Results plotted and saved.")
    
    print("Real robot control complete!")

if __name__ == "__main__":
    main()