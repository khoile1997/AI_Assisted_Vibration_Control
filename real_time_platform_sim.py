"""
real_time_platform_sim_enhanced.py

Enhanced Real-time simulator for a rectangular platform with 4 corner actuators.

NEW FEATURES:
- 3D visualization of platform motion in real-time
- Corner displacement graphs (x1, x2, x3, x4) correlated with motor currents
- Improved visualization layout with multiple subplots

Original Features:
- Small-angle roll (phi) and pitch (theta) dynamics plus vertical translation (z).
- Actuators modeled as velocity-dependent dampers where damping coefficient = k_c * current.
- Actuator current limits (saturation) and per-actuator force limit (F_max).
- Force saturation handled by clipping each actuator force.
- RK4 integrator for dynamics.
- Real-time interactive matplotlib plot updating live.
- TCP JSON control server (runs in background thread) so a separate program/process can:
    - set currents: {"cmd":"set_currents", "currents":[I1,I2,I3,I4]}
    - set state: {"cmd":"set_state", "state":{"phi":..,"theta":..,"z":..,"phi_dot":..,"theta_dot":..,"z_dot":..}}
    - set external loads: {"cmd":"set_external","force_z":..,"moment_x":..,"moment_y":..}
    - request state: {"cmd":"get_state"} -> returns current sensor reading JSON
    - set limits: {"cmd":"set_limits", "I_min":.., "I_max":.., "F_max":..}
    - other simple commands

Requirements:
- numpy, matplotlib
- Python 3.7+

Usage:
- Run this script. The server will listen (by default) on localhost:5005.
- Run a separate client (example provided below) to send commands to the simulator.
"""

import threading
import socketserver
import socket
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------
# Simulator core
# -------------------------
class PlatformSimulatorRT:
    def __init__(
        self,
        Lx=0.8,
        Ly=0.5,
        mass=10.0,
        I_x=0.8,
        I_y=0.9,
        k_c=0.8,
        viscous_b=np.array([0.02, 0.02]),  # rotational viscous [b_phi, b_theta]
        stiffness_k=np.array([0.0, 0.0]),  # rotational stiffness [k_phi, k_theta]
        b_z=1.0,                            # vertical viscous damping
        k_z=0.0,                            # vertical stiffness
        noise_std_sensors=0.0,
        I_limits=(-2.0, 2.0),               # (I_min, I_max) current saturation per motor
        F_max=100.0,                        # per-actuator force magnitude saturation (N)
    ):
        """
        Initialize the simulator.

        State vector (internal):
          phi, theta, z, phi_dot, theta_dot, z_dot

        Corner coordinates (x forward, y right) in meters, center at origin:
          corners: front-left, front-right, rear-left, rear-right
        """
        self.Lx = Lx
        self.Ly = Ly
        self.mass = mass
        self.I = np.array([I_x, I_y])
        self.k_c = float(k_c)
        self.b_rot = np.array(viscous_b, dtype=float)
        self.k_rot = np.array(stiffness_k, dtype=float)
        self.b_z = float(b_z)
        self.k_z = float(k_z)
        self.noise_std = float(noise_std_sensors)
        self.I_min, self.I_max = I_limits
        self.F_max = float(F_max)

        # corners in platform frame (x=forward, y=right):
        #   1: front-left   (+Lx/2, -Ly/2)
        #   2: front-right  (+Lx/2, +Ly/2)
        #   3: rear-left    (-Lx/2, -Ly/2)
        #   4: rear-right   (-Lx/2, +Ly/2)
        self.corners_flat = np.array([
            [+Lx / 2, -Ly / 2],
            [+Lx / 2, +Ly / 2],
            [-Lx / 2, -Ly / 2],
            [-Lx / 2, +Ly / 2]
        ])

        # State
        self.state = np.zeros(6)  # [phi, theta, z, phi_dot, theta_dot, z_dot]
        self.currents = np.zeros(4)
        self.external = {"force_z": 0.0, "moment_x": 0.0, "moment_y": 0.0}

    def reset(self, phi=0.0, theta=0.0, z=0.0, phi_dot=0.0, theta_dot=0.0, z_dot=0.0):
        self.state = np.array([phi, theta, z, phi_dot, theta_dot, z_dot], dtype=float)

    def set_currents(self, I):
        I = np.array(I, dtype=float).flatten()
        if I.shape[0] != 4:
            raise ValueError("currents must be array of size 4")
        # saturate
        I = np.clip(I, self.I_min, self.I_max)
        self.currents = I

    def set_state(self, state_dict):
        """
        Set simulator state from a dict with some or all of: phi, theta, z, phi_dot, theta_dot, z_dot
        """
        mapping = {"phi": 0, "theta": 1, "z": 2, "phi_dot": 3, "theta_dot": 4, "z_dot": 5}
        for k, idx in mapping.items():
            if k in state_dict:
                self.state[idx] = float(state_dict[k])

    def set_external(self, ext_dict):
        for k in ("force_z", "moment_x", "moment_y"):
            if k in ext_dict:
                self.external[k] = float(ext_dict[k])

    def set_limits(self, I_min=None, I_max=None, F_max=None):
        if I_min is not None:
            self.I_min = float(I_min)
        if I_max is not None:
            self.I_max = float(I_max)
        if F_max is not None:
            self.F_max = float(F_max)

    def get_state(self):
        """
        Return current state + currents as a dict (sensor readout).
        Add noise if noise_std > 0.
        """
        s = self.state.copy()
        if self.noise_std > 0:
            s += np.random.randn(6) * self.noise_std
        return {
            "state": {
                "phi": float(s[0]),
                "theta": float(s[1]),
                "z": float(s[2]),
                "phi_dot": float(s[3]),
                "theta_dot": float(s[4]),
                "z_dot": float(s[5]),
            },
            "currents": list(self.currents)
        }

    def get_corner_displacements(self):
        """
        Calculate vertical displacement of each corner relative to neutral position.
        
        Returns:
            x1, x2, x3, x4: vertical displacements of corners 1-4 (meters)
        """
        phi, theta, z = self.state[0], self.state[1], self.state[2]
        
        # Corner positions in platform frame
        corners = self.corners_flat  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        # Vertical displacement = z + (rotation effects)
        # For small angles: dz ≈ z - x*sin(theta) + y*sin(phi)
        # Or more precisely: dz ≈ z - x*theta + y*phi
        
        x1 = z - corners[0, 0] * theta + corners[0, 1] * phi
        x2 = z - corners[1, 0] * theta + corners[1, 1] * phi
        x3 = z - corners[2, 0] * theta + corners[2, 1] * phi
        x4 = z - corners[3, 0] * theta + corners[3, 1] * phi
        
        return x1, x2, x3, x4

    def compute_actuator_forces(self):
        """
        Compute forces from 4 actuators based on currents and corner velocities.
        Each actuator: F_i = -k_c * I_i * z_dot_i
        where z_dot_i is vertical velocity at corner i.
        """
        phi, theta, z, phi_dot, theta_dot, z_dot = self.state
        corners = self.corners_flat

        # Vertical velocities at corners (small angle approximation)
        # z_dot_i = z_dot - x_i*theta_dot + y_i*phi_dot
        z_dot_corners = np.array([
            z_dot - corners[i, 0] * theta_dot + corners[i, 1] * phi_dot
            for i in range(4)
        ])

        # Actuator forces: F_i = -k_c * I_i * z_dot_i
        F_raw = -self.k_c * self.currents * z_dot_corners

        # Saturate per-actuator force
        F = np.clip(F_raw, -self.F_max, self.F_max)
        return F

    def rhs(self, state_vec):
        """
        Right-hand side of dynamics: dstate/dt = rhs(state)
        state_vec = [phi, theta, z, phi_dot, theta_dot, z_dot]
        """
        phi, theta, z, phi_dot, theta_dot, z_dot = state_vec

        # Actuator forces
        F = self.compute_actuator_forces()

        # Total force and moments from actuators
        corners = self.corners_flat
        F_z_act = np.sum(F)
        M_x_act = np.sum(F * corners[:, 1])  # sum of F_i * y_i
        M_y_act = np.sum(F * corners[:, 0])  # sum of F_i * x_i

        # External loads
        F_z_ext = self.external["force_z"]
        M_x_ext = self.external["moment_x"]
        M_y_ext = self.external["moment_y"]

        # Equations of motion (small angle approximation)
        # Rotational
        ddphi = (M_x_act + M_x_ext - self.b_rot[0] * phi_dot - self.k_rot[0] * phi) / self.I[0]
        ddtheta = (M_y_act + M_y_ext - self.b_rot[1] * theta_dot - self.k_rot[1] * theta) / self.I[1]

        # Vertical
        ddz = (F_z_act + F_z_ext - self.b_z * z_dot - self.k_z * z) / self.mass

        dstate = np.array([phi_dot, theta_dot, z_dot, ddphi, ddtheta, ddz])
        return dstate

    def step(self, dt):
        """
        Integrate state forward by dt using RK4.
        """
        s = self.state
        k1 = self.rhs(s)
        k2 = self.rhs(s + 0.5 * dt * k1)
        k3 = self.rhs(s + 0.5 * dt * k2)
        k4 = self.rhs(s + dt * k3)
        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -------------------------
# TCP server for external control
# -------------------------
class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            try:
                line = self.rfile.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                cmd = json.loads(line)
                response = self.server.process_command(cmd)
                response_str = json.dumps(response) + "\n"
                self.wfile.write(response_str.encode("utf-8"))
            except Exception as e:
                err_resp = {"error": str(e)}
                self.wfile.write((json.dumps(err_resp) + "\n").encode("utf-8"))
                break


class ControlTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, simulator: PlatformSimulatorRT):
        super().__init__(server_address, RequestHandlerClass)
        self.simulator = simulator

    def process_command(self, cmd):
        """
        Process a single command dictionary and return a response dict.
        Supported commands:
          - set_currents: {"cmd":"set_currents","currents":[I1,I2,I3,I4]}
          - set_state: {"cmd":"set_state","state":{...}}
          - set_external: {"cmd":"set_external", ...}
          - get_state: {"cmd":"get_state"}
          - set_limits: {"cmd":"set_limits", ...}
        """
        try:
            c = cmd.get("cmd", "").lower()
            if c == "set_currents":
                currents = cmd.get("currents")
                if currents is None:
                    return {"error": "missing 'currents' field"}
                self.simulator.set_currents(currents)
                return {"ok": True, "currents": list(self.simulator.currents)}
            elif c == "set_state":
                state = cmd.get("state", {})
                self.simulator.set_state(state)
                return {"ok": True, "state": self.simulator.get_state()["state"]}
            elif c == "set_external":
                ext = {k: v for k, v in cmd.items() if k in ("force_z", "moment_x", "moment_y")}
                self.simulator.set_external(ext)
                return {"ok": True, "external": dict(self.simulator.external)}
            elif c == "get_state":
                return {"ok": True, "data": self.simulator.get_state()}
            elif c == "set_limits":
                self.simulator.set_limits(I_min=cmd.get("I_min"), I_max=cmd.get("I_max"), F_max=cmd.get("F_max"))
                return {"ok": True, "limits": {"I_min": self.simulator.I_min, "I_max": self.simulator.I_max, "F_max": self.simulator.F_max}}
            else:
                return {"error": f"unknown cmd '{c}'"}
        except Exception as e:
            return {"error": str(e)}

# -------------------------
# Enhanced Real-time plotting / main loop
# -------------------------
class RealTimeApp:
    def __init__(self, simulator: PlatformSimulatorRT, dt=0.01, host="127.0.0.1", port=5005):
        self.sim = simulator
        self.dt = float(dt)
        self.server = None
        self.server_thread = None
        self.host = host
        self.port = port
        self.running = False

        # data buffers for plotting
        self.window = 10.0  # seconds of data to show
        npoints = max(20, int(self.window / self.dt))
        self.times = np.linspace(-self.window, 0.0, npoints)
        self.phi_hist = np.zeros(npoints)
        self.theta_hist = np.zeros(npoints)
        self.z_hist = np.zeros(npoints)
        self.currents_hist = np.zeros((npoints, 4))
        self.x_hist = np.zeros((npoints, 4))  # Corner displacements

    def start_server(self):
        # start TCP server in background thread
        server = ControlTCPServer((self.host, self.port), ThreadedTCPRequestHandler, self.sim)
        self.server = server
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.server_thread = t
        print(f"Control server listening on {self.host}:{self.port}")

    def stop_server(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def run(self):
        # start server
        self.start_server()
        self.running = True

        # set up matplotlib figure with enhanced layout
        plt.style.use("seaborn-v0_8-darkgrid")
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid: 3D plot on left, time series on right
        gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
        
        # 3D visualization (spans all rows on left)
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        
        # Time series plots on right
        ax_angles = fig.add_subplot(gs[0, 1])
        ax_currents = fig.add_subplot(gs[1, 1], sharex=ax_angles)
        ax_displacements = fig.add_subplot(gs[2, 1], sharex=ax_angles)

        # === 3D Platform Visualization ===
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('Platform 3D Visualization', fontsize=14, fontweight='bold')
        
        # Set fixed limits for 3D plot
        ax_3d.set_xlim([-0.6, 0.6])
        ax_3d.set_ylim([-0.4, 0.4])
        ax_3d.set_zlim([-0.15, 0.15])
        
        # Initial platform corners
        platform_plot = None
        corner_scatters = []
        
        # === Angles Plot ===
        line_phi, = ax_angles.plot(self.times, self.phi_hist, 'b-', label='φ (roll)', linewidth=1.5)
        line_theta, = ax_angles.plot(self.times, self.theta_hist, 'r-', label='θ (pitch)', linewidth=1.5)
        ax_angles.set_ylabel('Angle (rad)', fontsize=10)
        ax_angles.legend(loc='upper right', fontsize=9)
        ax_angles.grid(True, alpha=0.3)
        ax_angles.set_title('Platform Angles', fontsize=11, fontweight='bold')

        # === Currents Plot ===
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        lines_currents = []
        for i in range(4):
            line, = ax_currents.plot(self.times, self.currents_hist[:, i], 
                                     color=colors[i], label=f'I{i+1}', linewidth=1.5)
            lines_currents.append(line)
        ax_currents.set_ylabel('Current (A)', fontsize=10)
        ax_currents.legend(loc='upper right', fontsize=9, ncol=2)
        ax_currents.grid(True, alpha=0.3)
        ax_currents.set_title('Motor Currents', fontsize=11, fontweight='bold')

        # === Corner Displacements Plot ===
        lines_displacements = []
        for i in range(4):
            line, = ax_displacements.plot(self.times, self.x_hist[:, i],
                                          color=colors[i], label=f'x{i+1}', linewidth=1.5)
            lines_displacements.append(line)
        ax_displacements.set_xlabel('Time (s)', fontsize=10)
        ax_displacements.set_ylabel('Displacement (m)', fontsize=10)
        ax_displacements.legend(loc='upper right', fontsize=9, ncol=2)
        ax_displacements.grid(True, alpha=0.3)
        ax_displacements.set_title('Corner Vertical Displacements', fontsize=11, fontweight='bold')

        fig.suptitle('Enhanced Platform Simulator - Real-time Visualization', 
                     fontsize=16, fontweight='bold')

        def update(frame):
            if not self.running:
                return

            # step simulator
            self.sim.step(self.dt)

            # get current state
            phi, theta, z = self.sim.state[:3]
            phi_dot, theta_dot, z_dot = self.sim.state[3:]
            currents = self.sim.currents
            x1, x2, x3, x4 = self.sim.get_corner_displacements()

            # roll data
            self.phi_hist = np.roll(self.phi_hist, -1)
            self.theta_hist = np.roll(self.theta_hist, -1)
            self.z_hist = np.roll(self.z_hist, -1)
            self.currents_hist = np.roll(self.currents_hist, -1, axis=0)
            self.x_hist = np.roll(self.x_hist, -1, axis=0)

            self.phi_hist[-1] = phi
            self.theta_hist[-1] = theta
            self.z_hist[-1] = z
            self.currents_hist[-1, :] = currents
            self.x_hist[-1, :] = [x1, x2, x3, x4]

            # === Update 3D Platform ===
            ax_3d.cla()
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title('Platform 3D Visualization', fontsize=14, fontweight='bold')
            ax_3d.set_xlim([-0.6, 0.6])
            ax_3d.set_ylim([-0.4, 0.4])
            ax_3d.set_zlim([-0.15, 0.15])
            
            # Calculate corner positions in 3D
            corners_2d = self.sim.corners_flat
            
            # Transform corners based on current angles
            # For small angles: rotation matrix approximation
            corners_3d = np.zeros((4, 3))
            for i in range(4):
                x_local = corners_2d[i, 0]
                y_local = corners_2d[i, 1]
                
                # Position after rotation (small angle approx)
                x_world = x_local
                y_world = y_local
                z_world = z - x_local * theta + y_local * phi
                
                corners_3d[i] = [x_world, y_world, z_world]
            
            # Draw platform as a filled polygon
            verts = [corners_3d[[0, 1, 3, 2]]]  # Correct order for filled quad
            platform_collection = Poly3DCollection(verts, alpha=0.7, facecolor='cyan', edgecolor='darkblue', linewidth=2)
            ax_3d.add_collection3d(platform_collection)
            
            # Draw corners as spheres
            for i, corner in enumerate(corners_3d):
                ax_3d.scatter(*corner, color=colors[i], s=100, marker='o', edgecolor='black', linewidth=1.5)
                ax_3d.text(corner[0], corner[1], corner[2]+0.02, f'  {i+1}', fontsize=10, fontweight='bold')
            
            # Draw vertical lines from corners to ground
            for corner in corners_3d:
                ax_3d.plot([corner[0], corner[0]], [corner[1], corner[1]], [corner[2], -0.15], 
                          'k--', alpha=0.3, linewidth=0.8)
            
            # Draw ground plane
            xx, yy = np.meshgrid(np.linspace(-0.6, 0.6, 3), np.linspace(-0.4, 0.4, 3))
            zz = np.full_like(xx, -0.15)
            ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
            
            # Set view angle
            ax_3d.view_init(elev=20, azim=45)

            # === Update Time Series Plots ===
            line_phi.set_ydata(self.phi_hist)
            line_theta.set_ydata(self.theta_hist)

            for i, line in enumerate(lines_currents):
                line.set_ydata(self.currents_hist[:, i])

            for i, line in enumerate(lines_displacements):
                line.set_ydata(self.x_hist[:, i])

            # Auto-scale y-axes
            ax_angles.relim()
            ax_angles.autoscale_view(scalex=False)
            ax_currents.relim()
            ax_currents.autoscale_view(scalex=False)
            ax_displacements.relim()
            ax_displacements.autoscale_view(scalex=False)

            return []

        ani = FuncAnimation(fig, update, interval=int(self.dt * 1000), blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

        self.running = False
        self.stop_server()


# -------------------------
# Client example
# -------------------------
CLIENT_EXAMPLE = """
import socket
import json

def send_command(cmd_dict, host='127.0.0.1', port=5005):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall((json.dumps(cmd_dict) + '\\n').encode('utf-8'))
    response = s.recv(4096).decode('utf-8')
    s.close()
    return json.loads(response)

# Example: set currents
resp = send_command({"cmd": "set_currents", "currents": [0.5, -0.3, 0.2, -0.1]})
print(resp)

# Example: get state
resp = send_command({"cmd": "get_state"})
print(resp)
"""

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":
    sim = PlatformSimulatorRT(
        Lx=0.8,
        Ly=0.5,
        mass=12.0,
        I_x=0.8,
        I_y=0.9,
        k_c=1.0,
        viscous_b=np.array([0.05, 0.05]),
        stiffness_k=np.array([0.0, 0.0]),
        b_z=2.0,
        k_z=0.0,
        noise_std_sensors=0.0,
        I_limits=(-2.0, 2.0),
        F_max=150.0,
    )

    # example initial condition
    sim.reset(phi=0.05, theta=-0.03, z=0.01, phi_dot=0.0, theta_dot=0.0, z_dot=0.0)
    # default currents
    sim.set_currents([0.0, 0.0, 0.0, 0.0])

    app = RealTimeApp(simulator=sim, dt=0.01, host="127.0.0.1", port=5005)
    print("Example client snippet to control the sim (run separately):")
    print(CLIENT_EXAMPLE)
    print("\n" + "="*70)
    print("ENHANCED FEATURES:")
    print("  - 3D visualization of platform showing real-time motion")
    print("  - Corner displacement graphs (x1, x2, x3, x4)")
    print("  - Correlation with motor currents (I1, I2, I3, I4)")
    print("="*70 + "\n")
    app.run()
