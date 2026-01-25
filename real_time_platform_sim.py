"""
real_time_platform_sim.py

Real-time simulator for a rectangular platform with 4 corner actuators.

Features:
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

        # corner coordinates (x,y) (front-left, front-right, rear-left, rear-right)
        hx = Lx / 2.0
        hy = Ly / 2.0
        self.corners = np.array(
            [
                [+hx, -hy],  # front-left
                [+hx, +hy],  # front-right
                [-hx, -hy],  # rear-left
                [-hx, +hy],  # rear-right
            ]
        )  # shape (4,2)

        # state: phi, theta, z, phi_dot, theta_dot, z_dot
        self.state = np.zeros(6)
        # commands / external inputs
        self.currents = np.zeros(4)  # commanded currents (A)
        self.external = {"force_z": 0.0, "moment_x": 0.0, "moment_y": 0.0}
        # locking for thread-safety
        self.lock = threading.RLock()

    def reset(self, phi=0.0, theta=0.0, z=0.0, phi_dot=0.0, theta_dot=0.0, z_dot=0.0):
        with self.lock:
            self.state[:] = np.array([phi, theta, z, phi_dot, theta_dot, z_dot], dtype=float)

    def set_currents(self, currents):
        with self.lock:
            arr = np.asarray(currents, dtype=float).reshape(4)
            # apply current limits
            arr = np.clip(arr, self.I_min, self.I_max)
            self.currents = arr

    def set_state(self, state_dict):
        with self.lock:
            # Accept partial dictionaries
            s = dict(state_dict)
            phi = s.get("phi", self.state[0])
            theta = s.get("theta", self.state[1])
            z = s.get("z", self.state[2])
            phi_dot = s.get("phi_dot", self.state[3])
            theta_dot = s.get("theta_dot", self.state[4])
            z_dot = s.get("z_dot", self.state[5])
            self.state[:] = np.array([phi, theta, z, phi_dot, theta_dot, z_dot], dtype=float)

    def set_external(self, ext_dict):
        with self.lock:
            for k in ("force_z", "moment_x", "moment_y"):
                if k in ext_dict:
                    self.external[k] = float(ext_dict[k])

    def set_limits(self, I_min=None, I_max=None, F_max=None):
        with self.lock:
            if I_min is not None:
                self.I_min = float(I_min)
            if I_max is not None:
                self.I_max = float(I_max)
            if F_max is not None:
                self.F_max = float(F_max)
            # enforce current clipping on stored currents
            self.currents = np.clip(self.currents, self.I_min, self.I_max)

    def _compute_force_terms(self, state, currents):
        """
        Compute per-actuator forces and the aggregated contributions:
         - forces: 4-element array (N) applied upward on platform (positive up)
         - sums needed for dynamics: S_c, S_cx, S_cy, S_cxx, S_cyy, S_cxy
        Includes actuator force saturation (per-actuator magnitude limited to F_max).
        """
        # map current to damping coefficient (positive)
        c = self.k_c * np.clip(currents, self.I_min, self.I_max)
        x = self.corners[:, 0]
        y = self.corners[:, 1]
        # velocities
        phi_dot = state[3]
        theta_dot = state[4]
        z_dot = state[5]

        # corner vertical velocity v_i = z_dot - phi_dot*y_i + theta_dot*x_i
        v = z_dot - phi_dot * y + theta_dot * x

        # ideal damping force (opposes velocity): F_i = -c_i * v_i
        F = -c * v

        # apply per-actuator force saturation
        if self.F_max is not None and self.F_max > 0.0:
            F = np.clip(F, -self.F_max, self.F_max)

        # For aggregated sums we need sums over c times coordinates, BUT after saturation
        # the effective "c" for aggregated linear terms is not straightforward if some actuators saturated.
        # We'll compute aggregated sums using the actual forces F and velocities to derive equivalent terms:
        # sum_F = sum F_i = -S_c * z_dot + S_cy * phi_dot - S_cx * theta_dot  (if no saturation)
        # Rather than try to invert, compute sums directly from F and known v, phi_dot, theta_dot:
        S_F = np.sum(F)
        # compute terms used in torque equations:
        S_yF = np.sum(y * F)   # contributes to roll moment
        S_xF = np.sum(x * F)   # contributes to pitch moment (with sign)
        # compute entries for damping matrix approximations when needed by linear terms:
        # We'll compute the sums of c*x^2 etc using original c (pre-saturation) for the linear parts but
        # for accuracy when saturation present, we will use the forces F to compute torques/moments exactly.
        # However we still return c-based sums for use in non-saturated linear approximation when required.
        S_c = np.sum(c)
        S_cx = np.sum(c * x)
        S_cy = np.sum(c * y)
        S_cxx = np.sum(c * x * x)
        S_cyy = np.sum(c * y * y)
        S_cxy = np.sum(c * x * y)

        return {
            "F": F,
            "v": v,
            "S_F": S_F,
            "S_xF": S_xF,
            "S_yF": S_yF,
            "S_c": S_c,
            "S_cx": S_cx,
            "S_cy": S_cy,
            "S_cxx": S_cxx,
            "S_cyy": S_cyy,
            "S_cxy": S_cxy,
        }

    def _derivatives(self, state, currents, external=None):
        """
        Given current state and currents, return state derivatives:
        state: [phi, theta, z, phi_dot, theta_dot, z_dot]
        returns: state_dot of same shape
        """
        if external is None:
            external = {"force_z": 0.0, "moment_x": 0.0, "moment_y": 0.0}

        phi, theta, z, phi_dot, theta_dot, z_dot = state
        terms = self._compute_force_terms(state, currents)

        # Rotational damping torque from actuators: compute actual torque directly using forces:
        # Mx = sum y_i * F_i
        # My = -sum x_i * F_i  (sign chosen consistent with earlier conventions)
        Mx_from_F = np.sum(self.corners[:, 1] * terms["F"])
        My_from_F = -np.sum(self.corners[:, 0] * terms["F"])

        # Include additional rotational viscous damping B_rot and stiffness K_rot:
        vel = np.array([phi_dot, theta_dot])
        torque_visc = self.b_rot * vel  # element-wise (diagonal)
        torque_stiff = self.k_rot * np.array([phi, theta])

        # Total rotational moments: from actuators + viscous + stiffness + external
        # Note: actuator contribution Mx_from_F, My_from_F already includes sign (forces are upward positive).
        # We collect RHS = actuator_moments - torque_visc - torque_stiff + external_moments
        Mx_total = Mx_from_F - torque_visc[0] - torque_stiff[0] + external.get("moment_x", 0.0)
        My_total = My_from_F - torque_visc[1] - torque_stiff[1] + external.get("moment_y", 0.0)

        # rotational accelerations:
        phi_dd = Mx_total / self.I[0]
        theta_dd = My_total / self.I[1]

        # Vertical dynamics:
        # sum forces on z (positive up) = sum F_i + (-b_z * z_dot) + (-k_z * z) + external force
        sum_F = terms["S_F"]
        z_dd = (sum_F - self.b_z * z_dot - self.k_z * z + external.get("force_z", 0.0)) / self.mass

        # Pack derivatives: [phi_dot, theta_dot, z_dot, phi_dd, theta_dd, z_dd]
        state_dot = np.array([phi_dot, theta_dot, z_dot, phi_dd, theta_dd, z_dd], dtype=float)
        return state_dot

    def step(self, dt):
        """
        Advance state by dt using RK4 with current commanded currents and external inputs.
        Returns sensor readings (with optional noise) as a dict.
        """
        with self.lock:
            s0 = self.state.copy()
            currents = self.currents.copy()
            external = dict(self.external)

        # RK4 using lambdas that close over currents/external
        k1 = self._derivatives(s0, currents, external)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, currents, external)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, currents, external)
        k4 = self._derivatives(s0 + dt * k3, currents, external)
        s_new = s0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        with self.lock:
            self.state = s_new

            # sensor readings
            phi, theta, z, phi_dot, theta_dot, z_dot = self.state
            if self.noise_std > 0.0:
                ang_noise = np.random.normal(0.0, self.noise_std, 3)
                rate_noise = np.random.normal(0.0, self.noise_std, 3)
            else:
                ang_noise = np.zeros(3)
                rate_noise = np.zeros(3)

            sensors = {
                "phi": phi + ang_noise[0],
                "theta": theta + ang_noise[1],
                "z": z + ang_noise[2],
                "phi_dot": phi_dot + rate_noise[0],
                "theta_dot": theta_dot + rate_noise[1],
                "z_dot": z_dot + rate_noise[2],
                "currents": self.currents.copy(),
                "external": dict(self.external),
                "timestamp": time.time(),
            }
        return sensors

    def get_state(self):
        with self.lock:
            s = {
                "state": {
                    "phi": float(self.state[0]),
                    "theta": float(self.state[1]),
                    "z": float(self.state[2]),
                    "phi_dot": float(self.state[3]),
                    "theta_dot": float(self.state[4]),
                    "z_dot": float(self.state[5]),
                },
                "currents": list(self.currents.copy()),
                "limits": {"I_min": self.I_min, "I_max": self.I_max, "F_max": self.F_max},
                "external": dict(self.external),
                "timestamp": time.time(),
            }
            return s

# -------------------------
# Simple TCP JSON server to control the sim from another program
# -------------------------
class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """
    Simple request handler expecting JSON commands terminated by newline.
    Responds with JSON terminated by newline.
    """
    def handle(self):
        # receive until connection closed
        data = b""
        # set small timeout so we don't block forever
        self.request.settimeout(0.2)
        try:
            while True:
                chunk = self.request.recv(4096)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pass  # proceed with whatever we got

        if not data:
            return

        # Accept multiple JSON objects separated by newline; process the first valid
        text = data.decode("utf-8").strip()
        responses = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                cmd = json.loads(line)
            except Exception as e:
                responses.append({"error": f"invalid json: {str(e)}"})
                continue

            # process command
            resp = self.server.process_command(cmd)
            responses.append(resp)

        # send all responses as JSON lines
        out_text = "\n".join(json.dumps(r) for r in responses) + "\n"
        try:
            self.request.sendall(out_text.encode("utf-8"))
        except Exception:
            pass


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
# Real-time plotting / main loop
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

        # set up matplotlib figure
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax1, ax2, ax3 = axs

        # lines
        (line_phi,) = ax1.plot(self.times, self.phi_hist, label="phi (rad)")
        (line_theta,) = ax1.plot(self.times, self.theta_hist, label="theta (rad)")
        ax1.set_ylabel("angle [rad]")
        ax1.legend(loc="upper right")

        (line_phi_dot,) = ax2.plot(self.times, np.zeros_like(self.times), label="phi_dot")
        (line_theta_dot,) = ax2.plot(self.times, np.zeros_like(self.times), label="theta_dot")
        (line_z_dot,) = ax2.plot(self.times, np.zeros_like(self.times), label="z_dot")
        ax2.set_ylabel("rates [rad/s, m/s]")
        ax2.legend(loc="upper right")

        line_currents = []
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i in range(4):
            ln, = ax3.plot(self.times, self.currents_hist[:, i], label=f"I{i+1}", color=colors[i])
            line_currents.append(ln)
        ax3.set_ylabel("current [A]")
        ax3.set_xlabel("time [s]")
        ax3.legend(loc="upper right")

        plt.tight_layout()

        last_time = time.time()

        def update(frame):
            nonlocal last_time
            # advance simulation enough steps to catch up real time or run at fixed dt
            now = time.time()
            elapsed = now - last_time
            # Limit step count to avoid spiral
            max_steps = int(min(10, max(1, elapsed / self.dt)))
            for _ in range(max_steps):
                sensors = self.sim.step(self.dt)
                last_time = time.time()

                # append to history buffers
                self.times = np.append(self.times[1:], sensors["timestamp"] - now)  # relative times
                self.phi_hist = np.append(self.phi_hist[1:], sensors["phi"])
                self.theta_hist = np.append(self.theta_hist[1:], sensors["theta"])
                self.z_hist = np.append(self.z_hist[1:], sensors["z"])
                self.currents_hist = np.vstack((self.currents_hist[1:, :], sensors["currents"]))

            # update plotted data
            line_phi.set_ydata(self.phi_hist)
            line_theta.set_ydata(self.theta_hist)
            line_phi_dot.set_ydata(np.gradient(self.phi_hist, self.times + 1e-12))  # numerical derivative approximate
            line_theta_dot.set_ydata(np.gradient(self.theta_hist, self.times + 1e-12))
            line_z_dot.set_ydata(np.gradient(self.z_hist, self.times + 1e-12))
            for i in range(4):
                line_currents[i].set_ydata(self.currents_hist[:, i])

            # keep x-limits fixed
            ax1.relim(); ax1.autoscale_view(scalex=False, scaley=True)
            ax2.relim(); ax2.autoscale_view(scalex=False, scaley=True)
            ax3.relim(); ax3.autoscale_view(scalex=False, scaley=True)
            return [line_phi, line_theta, line_phi_dot, line_theta_dot, line_z_dot] + line_currents

        ani = FuncAnimation(plt.gcf(), update, interval=int(self.dt * 1000), blit=False)
        try:
            print("Starting real-time plot. Close the plot window to exit.")
            plt.show()
        finally:
            print("Shutting down server and exiting...")
            self.stop_server()
            self.running = False

# -------------------------
# Example client usage (separate program)
# -------------------------
CLIENT_EXAMPLE = r"""
# Example control client to send commands to the simulator (run separately)
import socket, json

HOST = "127.0.0.1"
PORT = 5005

def send_cmd(cmd):
    s = socket.create_connection((HOST, PORT), timeout=1.0)
    txt = json.dumps(cmd) + "\n"
    s.sendall(txt.encode('utf-8'))
    # read response
    resp = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
    except Exception:
        pass
    s.close()
    print(resp.decode('utf-8'))

# set currents
send_cmd({"cmd":"set_currents", "currents":[1.0, 1.0, 0.5, 0.5]})

# get state
send_cmd({"cmd":"get_state"})

# apply an external push upward on platform
send_cmd({"cmd":"set_external", "force_z": 20.0})

# set actuator limits
send_cmd({"cmd":"set_limits", "I_min": -3.0, "I_max": 3.0, "F_max": 200.0})

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
    app.run()
