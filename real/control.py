import socket
import struct
import sys
import os
import threading
import time
from matplotlib import animation, pyplot as plt
import numpy as np
from zaber_motion.ascii import Connection
from zaber_motion import Units

# Local imports
import src.config as config
from src.pressure_loader import PressureLoader
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
from VLM import VLM, calculate_circle_through_points

class ThreadedRobotController:
    def __init__(self, simulation_params):

        # Save simulation parameters
        self.simulation_params = simulation_params

        # Control flags
        self.quit = False
        self.started = False
        self.quit_lock = threading.Lock()

        # Control variables
        self.offsets = simulation_params.get('pressure_offsets', [0.0, 0.0, 0.0])
        self.initial_pos = config.initial_pos
        self.max_pos = self.initial_pos + config.max_stroke
        self.default_target = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Default target if VLM fails
        
        # Control filter parameters
        self.filter_alpha = 0.8  # Low-pass filter coefficient (0 < alpha < 1, higher = less filtering)
        self.filtered_control = [self.initial_pos, self.initial_pos, self.initial_pos]  # Initialize with initial position
        
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
        self.keyboard_thread = None
        
        # History for plotting
        self.history_x = []
        self.history_u = []
        self.history_x_target = []
        self.history_mpc_times = []

        # Connect to devices
        self.connect_devices()
        # input("Press Enter to move to initial position...")
        # self.move_to_init()
        # print("Moved to initial position.")
        # input(f"Press Enter to load pressure")
        # self.add_offsets()

    def connect_devices(self):
        """
        Connect to the Zaber devices and initialize the oscilloscopes.
        """
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()
        device_list = connection.detect_devices()
        print("Found {} devices.".format(len(device_list)))
        print(device_list)

        # Get the axis
        self.axis_1 = device_list[0].get_axis(1)
        self.axis_2 = device_list[1].get_axis(1)
        self.axis_3 = device_list[2].get_axis(1)

        # Initialize oscilloscopes
        self.scope_1 = device_list[0].oscilloscope
        self.scope_2 = device_list[1].oscilloscope
        self.scope_3 = device_list[2].oscilloscope
        print(f'Oscilloscope 1 can store {self.scope_1.get_max_buffer_size()} samples.')
        print(f'Oscilloscope 2 can store {self.scope_2.get_max_buffer_size()} samples.')
        print(f'Oscilloscope 3 can store {self.scope_3.get_max_buffer_size()} samples.')
        return
    
    def move(self, control):
        # Apply low-pass filter to smooth control signals
        for i in range(len(control)):
            self.filtered_control[i] = self.filter_alpha * control[i] + (1 - self.filter_alpha) * self.filtered_control[i]
        
        # Check control limits using filtered control
        epsilon = 1e-3
        for i, c in enumerate(self.filtered_control):
            if c + self.offsets[i] < self.initial_pos + self.offsets[i] - epsilon or c + self.offsets[i] > self.max_pos + self.offsets[i] + epsilon:
                print(f"Filtered control {i} out of limits: {c}, moving to initial position instead.")
                self.move([self.initial_pos + self.offsets[0], self.initial_pos + self.offsets[1], self.initial_pos + self.offsets[2]])
                return
        # print(f"Moving to: {self.filtered_control}", flush=True)
        self.axis_1.move_absolute(self.filtered_control[0] + self.offsets[0],  Units.LENGTH_MILLIMETRES, False)
        self.axis_2.move_absolute(self.filtered_control[1] + self.offsets[1],  Units.LENGTH_MILLIMETRES, False)
        self.axis_3.move_absolute(self.filtered_control[2] + self.offsets[2],  Units.LENGTH_MILLIMETRES, False)
        return

    def move_to_init(self):
        """Move to the initial position"""
        self.move([self.initial_pos, self.initial_pos, self.initial_pos])

    def add_offsets(self):
        "Move to init plus offsets"
        self.move([self.initial_pos + self.offsets[0], self.initial_pos + self.offsets[1], self.initial_pos + self.offsets[2]])

    def set_quit_flag(self):
        """Thread-safe setter for quit flag"""
        with self.quit_lock:
            self.quit = True
    
    def get_quit_flag(self):
        """Thread-safe getter for quit flag"""
        with self.quit_lock:
            return self.quit

    def keyboard_input_thread(self):
        """Dedicated thread for keyboard input monitoring"""
        print("Keyboard input thread started. Press 'q' + Enter to quit gracefully.")
        try:
            while not self.get_quit_flag():
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        print("Quit command received from keyboard.")
                        self.set_quit_flag()
                        break
                except EOFError:
                    # Handle EOF gracefully
                    break
                except Exception as e:
                    print(f"Keyboard input error: {e}")
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard thread interrupted.")
        print("Keyboard input thread stopped.")

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
            
    def get_vlm_target_safe(self, mode):
        """Thread-safe getter for current VLM target"""
        with self.trajectory_lock:
            if mode == 'trajectory_tracking':
                if self.vlm_trajectory is not None:
                    if self.vlm_trajectory_index < len(self.vlm_trajectory):
                        target = self.vlm_trajectory[self.vlm_trajectory_index].copy()
                        self.vlm_trajectory_index += 1
                        return target
                    else:
                        return self.vlm_trajectory[-1].copy()
            if mode == 'set_point_regulation':
                if self.vlm_trajectory is not None:
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
        target_dt = self.simulation_params['vlm_dt']
        saved_first_image = False
        while not self.get_quit_flag():
            cycle_start_time = time.time()
            
            try:
                # Get current state (thread-safe)
                current_state = self.get_current_state_safe()
                if current_state is None:
                    print("VLM: Waiting for initial state...")
                    time.sleep(0.1)
                    continue
                
                # Get robot base and body from tracker 
                robot_base = self.tracker.get_current_base()
                robot_body = self.tracker.get_current_body()
                
                # Generate scene image with robot base and body
                try:
                    scene_image = self.vlm.ingest_info_real(current_state, robot_base, robot_body)
                except Exception as e:
                    print(f"Error generating scene image: {e}")

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
        target_dt = self.simulation_params['mpc_dt']  
        
        # Default target
        x_target = self.default_target.copy()
        
        while not self.get_quit_flag():
            cycle_start_time = time.time()
            
            try:
                # Get current state (thread-safe)
                current_state = self.get_current_state_safe()
                if current_state is None:
                    print("MPC: Waiting for initial state...")
                    time.sleep(0.1)
                    continue
                
                # Update target from VLM if available
                vlm_target = self.get_vlm_target_safe(mode='set_point_regulation')
                # print(f"VLM target: {vlm_target}")
                if vlm_target is not None:
                    x_target = vlm_target
                    # Add a small delta to the x of the target to stay below it
                    x_target[0] += 0.2  # Small delta
                
                # Compute MPC control
                start_mpc_time = time.time()
                x_target = self.default_target.copy()
                x_target = np.array([1.16575358, -0.95273664,  0.33866089, 0.0, 0.0, 0.0]) # brown target
                # x_target = np.array([1.19999521, 0.88552074, 0.38695709, 0.0, 0.0, 0.0]) # green target
                # x_target = np.array([0.13007123, -0.18891038, -1.35748895, 0.0, 0.0, 0.0]) # lightblue target
                x_target[0] += 0.2  # Small delta
                u_mpc = self.mpc.step(x_target, current_state)
                dist_to_target = np.linalg.norm(current_state - x_target)
                if self.started: print(f"MPC: u* = {u_mpc}, Pos. Distance to target: {dist_to_target:.4f}", end='\r', flush=True)
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
            self.move(control)

    def send_quit_signal(self):
        """Send a quit signal to the tracker via UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = struct.pack('?', True)
        sock.sendto(data, (config.UDP_IP, config.UDP_QUIT_TRACK_PORT))
        sock.close()
        print("Sent quit signal to tracker.")

    def quit_application(self):
        """Quit the application and clean up resources."""
        print("Quitting application...")
        self.set_quit_flag()

        # Stop all threads
        print("Stopping VLM...")
        if self.vlm:
            self.vlm.stop()
        
        print("Sending quit signal to tracker...")
        self.send_quit_signal()
        
        # Join all threads with timeout
        threads_to_join = [
            ("keyboard", self.keyboard_thread),
            ("VLM", self.vlm_thread),
            ("MPC", self.mpc_thread),
            ("tracker", self.tracker_thread)
        ]
        
        for thread_name, thread in threads_to_join:
            if thread and thread.is_alive():
                print(f"Stopping {thread_name} thread...")
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"Warning: {thread_name} thread did not stop within timeout")
                else:
                    print(f"{thread_name} thread joined.")
        
        print("Application quit complete.")

def main():    
    # Simulation settings
    FINAL_TIME = 10.0
    CONTROL_MODE = "vlm"  # set point regulation, "tt" for trajectory tracking, "vlm" for VLM control
    APPROXIMATION_ORDER = 2

    # Real robot parameters
    simulation_params = {
        'mpc_dt': 0.1,       # 10Hz
        'vlm_dt': 10.0,      # 1Hz
        'control_mode': CONTROL_MODE,
        'approximation_order': APPROXIMATION_ORDER,
        'final_time': FINAL_TIME,
        'offsets': [0.0, 0.0, 0.0]
    }

    # Load pressure 
    offsets = []
    pressure_loader = PressureLoader(save_offsets=False)
    offsets = pressure_loader.load_pressure()
    simulation_params['offsets'] = offsets
    print(f"Loaded pressure offsets: {offsets}\n")

    print("Initializing Real Robot PSS-VLMPC...")
    
    # Create controller instance
    controller = ThreadedRobotController(simulation_params)

    # Set up the figure for 3D plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("PSS-VLMPC Live Plot")
    ax.view_init(elev=-50, azim=-102.5, roll=-80)
    
    # Create scatter plots for visualization
    base_scatter = ax.scatter([0], [0], [0], s=100, c="yellow", label="Base")
    tip_scatter = ax.scatter([2.0], [0], [0], s=250, c="red", label="Current Tip")
    # Draw a line between base and tip
    body_arc = ax.plot([0, 2.0], [0, 0], [0, 0], c="blue", label="Body Arc", linewidth=5)

    # Draw targets
    targets = {
        'brown': np.array([1.16575358, -0.95273664,  0.33866089]),
        'green': np.array([1.19999521, 0.88552074, 0.38695709]),
        'lightblue': np.array([0.13007123, -0.18891038, -1.35748895]),
    }

    for target_name, target_pos in targets.items():
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c=target_name, s=80, label=f'{target_name} target')

    # Add legend
    ax.legend()

    def control_thread(controller):
        try:
            # Initialize Tracker
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
            if controller.simulation_params['control_mode'] == "vlm":
                print("\nInitializing VLM...")
                controller.vlm = VLM(
                    sim=False,
                    vlm_dt=controller.simulation_params['vlm_dt'],
                    mpc_dt=controller.simulation_params['mpc_dt'],
                    backend="gemini",
                    model_name="gemini-2.5-pro",
                    web_ui=True
                )

                if not controller.vlm.check_server():
                    print("Warning: VLM server not running! Switching to set point regulation mode.")
                    controller.simulation_params['control_mode'] = 'spr'
                else:
                    print("VLM server connected successfully!")
                    controller.vlm.start_input_thread()
                    
                    # Start VLM worker thread
                    controller.vlm_thread = threading.Thread(target=controller.vlm_worker_thread)
                    controller.vlm_thread.start()
                    print("VLM worker thread started.")

            # Initialize MPC controller
            print("\nInitializing MPC Controller...")
            controller.mpc = MPCController(nn_approximation_order=APPROXIMATION_ORDER)

            # Start MPC worker thread
            controller.mpc_thread = threading.Thread(target=controller.mpc_worker_thread)
            controller.mpc_thread.start()
            print("MPC worker thread started.")
            
            # Start keyboard input thread
            controller.keyboard_thread = threading.Thread(target=controller.keyboard_input_thread)
            controller.keyboard_thread.start()
            
            # Main loop for high-frequency operations
            print("\nStarting main control loop...")
            main_loop_dt = 0.01  # 100Hz for robot command sending
            controller.started = True
            while not controller.get_quit_flag():
                loop_start_time = time.time()
                
                # Update state from tracker
                controller.update_state_from_tracker()
                
                # Send robot command at high frequency
                controller.send_robot_command()
                
                # Time-compensated sleep for main loop
                elapsed = time.time() - loop_start_time
                remaining_time = main_loop_dt - elapsed
                if remaining_time > 0:
                    time.sleep(remaining_time)

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
        except Exception as e:
            print(f"Error during control: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always call quit_application to clean up
            controller.quit_application()
    
    # Start control thread
    ctrl_thread = threading.Thread(target=control_thread, args=(controller,))
    ctrl_thread.start()

    # Animation function to update the plot
    def animate(frame):
        try:
            # Get current state from robot tracker
            if controller.tracker is None:
                base = None
                tip = None
                body = None
                if frame % 100 == 0:  # Print debug info every 100 frames
                    print("Animation: Tracker not initialized yet")
            else:
                base = controller.tracker.get_current_base()
                tip = controller.tracker.get_current_tip()
                body = controller.tracker.get_current_body()

            # Initialize position arrays
            base_x, base_y, base_z = [], [], []
            tip_x, tip_y, tip_z = [], [], []
            body_x, body_y, body_z = [], [], []

            # Process base position
            if base is not None:
                base = base.ravel()
                base_x, base_y, base_z = [base[0]], [base[1]], [base[2]]

            # Process tip position
            if tip is not None:
                tip = tip.ravel()
                tip_x, tip_y, tip_z = [tip[0]], [tip[1]], [tip[2]]

            # Process body position
            if body is not None:
                body = body.ravel()
                body_x, body_y, body_z = [body[0]], [body[1]], [body[2]]

            # Clear previous lines (including previous circle arcs)
            for line in list(ax.get_lines()):
                if hasattr(line, 'get_label') and line.get_label() != "Planned Trajectory":
                    line.remove()

            # Draw circle arc for the body if we have all three points
            if tip_x and body_x:  # Both tip and body positions available
                try:
                    # Points: base (origin), body, tip
                    base_point = np.array([0, 0, 0])
                    body_point = np.array([body_x[0], body_y[0], body_z[0]])
                    tip_point = np.array([tip_x[0], tip_y[0], tip_z[0]])
                    
                    # Calculate circle through the three points
                    calculated_points = calculate_circle_through_points(body_point, tip_point, base_point, num_points=50)

                    if calculated_points is not None:
                        circle_points = calculated_points
                        body_arc = ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], c="blue", label="Body Arc", linewidth=5)

                except Exception as e:
                    # If circle calculation fails, use default line interpolation
                    if frame % 100 == 0:
                        print(f"Circle calculation failed: {e}")
                    # Draw a straight line from base 0,0,0 to tip
                    ax.plot([0, tip_x[0]], [0, tip_y[0]], [0, tip_z[0]], c="blue", label="Body Arc", linewidth=5)

            # Garbage collection every 50 frames
            if frame % 50 == 0:
                import gc
                gc.collect()

            # Update scatter plot positions
            base_scatter._offsets3d = ([0], [0], [0])  # Origin point for base
            tip_scatter._offsets3d = (tip_x, tip_y, tip_z)
            return

        except Exception as e:
            print(f"Animation error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Create the animation
    anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, interval=500)
    
    # Show the plot after controller has started
    while not controller.started:
        time.sleep(0.01)
    plt.tight_layout()
    plt.show()
    
    try:
        # Wait for the control thread to complete
        ctrl_thread.join()
    except KeyboardInterrupt:
        print("\nMain thread interrupted by user")
        controller.quit_application()
        ctrl_thread.join(timeout=5.0)

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