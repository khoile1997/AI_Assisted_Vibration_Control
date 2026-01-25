# controller_reinforcement_agent.py
"""
Reinforcement learning controller that reads initial vibration states from
'vibration_data.csv' and controls the 4 actuator currents of a running
real_time_platform_sim.py (which must be running and exposing the TCP JSON
control interface, default host=127.0.0.1 port=5005).

Highlights:
- Uses a simple policy-gradient (REINFORCE) agent with a small MLP policy (PyTorch).
- Episodes: each episode is initialized from one row of vibration_data.csv
  (columns expected: phi, theta, phi_dot, theta_dot). The agent then sends
  currents repeatedly until the platform is recovered (tilt & rates below
  thresholds) or a max episode time is reached.
- Reward (episodic) is computed as:
    reward = -(T * w_t + E * w_a + V * w_v) + B * I
  where:
    - T: time-to-recovery (s). If never recovered, T = max_episode_time.
    - E: actuation effort (approximated as integral of sum(I_i^2) dt)
    - V: total time during episode where max(|phi_dot|,|theta_dot|) > motion_rate_threshold
    - I: indicator (1 if T < t_thresh AND final tilt rates < rate_thresh, else 0)
    - w_t, w_a, w_v, B, thresholds are configurable hyperparameters.
- Communicates with simulator via the same JSON/TCP API used by the sim:
    - {"cmd":"set_currents", "currents":[...]}
    - {"cmd":"set_state", "state":{...}}
    - {"cmd":"get_state"} -> returns full state
    - {"cmd":"set_limits", ...} etc.
- Training is synchronous and episodic: for each episode the agent acts at a fixed
  control interval and collects one episodic reward; policy is updated with REINFORCE.
- This script is intentionally lightweight and dependency-minimal (requires numpy, pandas, torch).

Usage (high-level):
1) Start real_time_platform_sim.py (it runs a control server and a sim loop).
2) Run this agent script to train or evaluate:
     python controller_reinforcement_agent.py --mode train
   or
     python controller_reinforcement_agent.py --mode eval
3) Monitor printed logs. Model weights can be saved/loaded.

Notes / assumptions:
- vibration_data.csv must have at least columns: phi, theta, phi_dot, theta_dot
  (in radians and rad/s). If more columns exist they are ignored.
- The simulator must be reachable at --host and --port.
- Currents are clipped by the sim to its configured limits.
- The agent action outputs currents for four motors (array length 4).
- This implementation keeps the reward episodic (single reward at episode end)
  and applies that same return to each timestep's log-probabilities (standard REINFORCE).
"""

import argparse
import socket
import json
import time
import threading
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Simulator TCP client
# -----------------------
class SimClient:
    def __init__(self, host="127.0.0.1", port=5005, timeout=1.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _send_cmd(self, cmd_dict):
        txt = json.dumps(cmd_dict) + "\n"
        resp_text = ""
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall(txt.encode("utf-8"))
                # read response until socket closed / timeout
                s.settimeout(self.timeout)
                resp = b""
                while True:
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    resp += chunk
                resp_text = resp.decode("utf-8").strip()
        except Exception as e:
            raise RuntimeError(f"SimClient comms error: {e}")
        # Try to parse JSON lines; return the first JSON-decoded object
        if not resp_text:
            return None
        try:
            # may be one or more lines
            lines = [ln for ln in resp_text.splitlines() if ln.strip()]
            parsed = [json.loads(ln) for ln in lines]
            return parsed[0] if parsed else None
        except Exception:
            # return raw text if JSON parse fails
            return resp_text

    def set_currents(self, currents):
        return self._send_cmd({"cmd": "set_currents", "currents": list(map(float, currents))})

    def get_state(self):
        return self._send_cmd({"cmd": "get_state"})

    def set_state(self, state_dict):
        return self._send_cmd({"cmd": "set_state", "state": state_dict})

    def set_limits(self, I_min=None, I_max=None, F_max=None):
        cmd = {"cmd": "set_limits"}
        if I_min is not None:
            cmd["I_min"] = float(I_min)
        if I_max is not None:
            cmd["I_max"] = float(I_max)
        if F_max is not None:
            cmd["F_max"] = float(F_max)
        return self._send_cmd(cmd)

# -----------------------
# Policy network (small)
# -----------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, hidden_sizes=(64, 64), action_dim=4, init_log_std=-1.0, i_min=-2.0, i_max=2.0):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        # We use a learned log_std per action (diagonal gaussian)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
        # bounds for scaling tanh outputs to actual current range
        self.i_min = float(i_min)
        self.i_max = float(i_max)

    def forward(self, obs):
        """
        obs: torch tensor shape (batch, obs_dim) or (obs_dim,)
        Returns: mean (action space), log_std
        """
        x = self.net(obs)
        mean = self.mean_head(x)
        return mean, self.log_std

    def sample_action(self, obs_np, deterministic=False):
        """
        obs_np: numpy array (obs_dim,) or (batch, obs_dim)
        returns: actions (numpy, shape (4,)), log_prob (torch scalar), and torch tensors involved for training
        """
        single = False
        if obs_np.ndim == 1:
            obs = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
            single = True
        else:
            obs = torch.from_numpy(obs_np.astype(np.float32))
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        if deterministic:
            z = mean
            logp = None
        else:
            eps = torch.randn_like(mean)
            z = mean + eps * std
            # compute log prob (diagonal gaussian)
            var = std.pow(2)
            logp = -0.5 * (((z - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
            logp = logp.sum(dim=1)
        # pass through tanh to get bounded in [-1,1], then scale to [i_min, i_max]
        tanh_z = torch.tanh(z)
        action = tanh_z * ((self.i_max - self.i_min) / 2.0) + (self.i_max + self.i_min) / 2.0
        if single:
            action_np = action.detach().cpu().numpy()[0]
            logp_val = None if logp is None else logp.detach().cpu().numpy()[0]
            return action_np, logp_val, (mean, log_std, z)
        else:
            return action.detach().cpu().numpy(), None if logp is None else logp.detach().cpu().numpy(), (mean, log_std, z)

# -----------------------
# Reinforcement training
# -----------------------
class REINFORCE_Agent:
    def __init__(
        self,
        sim_client: SimClient,
        csv_path: str,
        dt_control=0.05,
        max_episode_time=8.0,
        angle_thresh=0.01,        # rad
        rate_thresh=0.05,        # rad/s
        motion_rate_thresh=0.2,  # rad/s (for V term)
        w_t=1.0,
        w_a=0.1,
        w_v=0.5,
        B=50.0,
        recovery_time_threshold=1.0,  # s for bonus condition
        lr=1e-3,
        gamma=0.99,
        device="cpu",
        i_min=-2.0,
        i_max=2.0,
    ):
        self.sim = sim_client
        self.dt = dt_control
        self.max_episode_time = float(max_episode_time)
        self.angle_thresh = float(angle_thresh)
        self.rate_thresh = float(rate_thresh)
        self.motion_rate_thresh = float(motion_rate_thresh)
        self.w_t = float(w_t)
        self.w_a = float(w_a)
        self.w_v = float(w_v)
        self.B = float(B)
        self.recovery_time_threshold = float(recovery_time_threshold)
        self.gamma = float(gamma)
        self.device = device

        # load CSV with initial conditions
        df = pd.read_csv(csv_path)
        # Expect columns 'phi','theta','phi_dot','theta_dot'; allow case-insensitive matching
        cols = {c.lower(): c for c in df.columns}
        required = ["phi", "theta", "phi_dot", "theta_dot"]
        for r in required:
            if r not in cols:
                raise ValueError(f"vibration_data.csv missing required column '{r}' (case-insensitive)")
        # keep as numpy array of shape (N,4)
        self.init_states = df[[cols["phi"], cols["theta"], cols["phi_dot"], cols["theta_dot"]]].to_numpy(dtype=float)
        self.n_init = self.init_states.shape[0]

        # policy net and optimizer
        self.policy = PolicyNet(obs_dim=4, hidden_sizes=(64, 64), action_dim=4, i_min=i_min, i_max=i_max).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def is_recovered(self, phi, theta, phi_dot, theta_dot):
        cond_angles = (abs(phi) <= self.angle_thresh) and (abs(theta) <= self.angle_thresh)
        cond_rates = (abs(phi_dot) <= self.rate_thresh) and (abs(theta_dot) <= self.rate_thresh)
        return cond_angles and cond_rates

    def run_episode(self, init_state, train=True, exploration_std_scale=1.0):
        """
        Run one episode starting from init_state = (phi,theta,phi_dot,theta_dot).
        Returns:
          - episodic_info: dict with T (time to recovery), E (effort), V (time above motion rate), indicator I, total_reward
          - trajectory: list of (obs, action, logp)
        """
        # set initial state in sim (z,z_dot left at 0)
        phi0, theta0, phi_dot0, theta_dot0 = map(float, init_state)
        self.sim.set_state({"phi": phi0, "theta": theta0, "phi_dot": phi_dot0, "theta_dot": theta_dot0, "z": 0.0, "z_dot": 0.0})
        # Clear currents initially
        self.sim.set_currents([0.0, 0.0, 0.0, 0.0])
        time.sleep(0.02)  # give sim a small moment to apply

        t = 0.0
        recovered_at = None
        E = 0.0
        V_time = 0.0
        trajectory = []  # list of (obs_np, action_np, logp_torch)
        last_obs = None

        # Warm read to have initial observation
        state_resp = self.sim.get_state()
        if not state_resp or "data" not in state_resp:
            # server may return state directly or wrapped; try to handle both.
            # If direct dict with 'state', use it.
            if isinstance(state_resp, dict) and "state" in state_resp:
                obs_s = state_resp["state"]["state"] if "state" in state_resp["state"] else state_resp["state"]
            else:
                # try directly response fields
                obs_s = state_resp.get("state") if isinstance(state_resp, dict) else None
        # We will fetch the canonical structure by another get_state with robust parsing below.

        while t < self.max_episode_time:
            # get current state
            resp = self.sim.get_state()
            if resp is None:
                raise RuntimeError("Failed to get state from simulator.")
            # resp structure: {"ok": True, "data": {...}} or direct data if server returned direct
            if isinstance(resp, dict) and resp.get("ok") and "data" in resp:
                sdata = resp["data"]
                # sdata has 'state' dict and 'currents', etc.
                sd = sdata["state"]
                phi, theta, z = sd["phi"], sd["theta"], sd["z"]
                phi_dot, theta_dot, z_dot = sd["phi_dot"], sd["theta_dot"], sd["z_dot"]
                currents_now = resp["data"].get("currents", [0, 0, 0, 0])
            elif isinstance(resp, dict) and "state" in resp:
                sd = resp["state"]
                phi, theta, z = sd.get("phi", 0.0), sd.get("theta", 0.0), sd.get("z", 0.0)
                phi_dot, theta_dot, z_dot = sd.get("phi_dot", 0.0), sd.get("theta_dot", 0.0), sd.get("z_dot", 0.0)
                currents_now = resp.get("currents", [0, 0, 0, 0])
            else:
                # fallback: try to parse common keys in top-level
                d = resp if isinstance(resp, dict) else {}
                phi = float(d.get("phi", 0.0))
                theta = float(d.get("theta", 0.0))
                phi_dot = float(d.get("phi_dot", 0.0))
                theta_dot = float(d.get("theta_dot", 0.0))
                currents_now = d.get("currents", [0, 0, 0, 0])
                z = float(d.get("z", 0.0))
                z_dot = float(d.get("z_dot", 0.0))

            obs = np.array([phi, theta, phi_dot, theta_dot], dtype=np.float32)
            last_obs = obs.copy()

            # check recovery
            if recovered_at is None and self.is_recovered(phi, theta, phi_dot, theta_dot):
                recovered_at = t
                # do not break immediately; allow the loop to register final state & energy
                # but you could break here to shorten episode; we will break shortly.

            # compute V term accumulation (if tilt rate above motion threshold)
            if max(abs(phi_dot), abs(theta_dot)) > self.motion_rate_thresh:
                V_time += self.dt

            # select action (currents)
            # add exploration by sampling from Gaussian centered on policy mean if training
            with torch.no_grad():
                action_np, logp_val, _ = self.policy.sample_action(obs, deterministic=(not train))
            # if training, add extra noise scaling factor (to encourage exploration)
            if train and exploration_std_scale != 0.0:
                # Gaussian noise in action space after scaling to limits
                noise = np.random.normal(scale=0.05 * exploration_std_scale, size=action_np.shape)
                action_np = np.clip(action_np + noise, self.policy.i_min, self.policy.i_max)

            # send currents to sim
            self.sim.set_currents(action_np.tolist())

            # accumulate effort: approximate E += sum(I^2) * dt
            E += (np.square(action_np).sum()) * self.dt

            # record trajectory info for training
            # We need log probability for REINFORCE; compute logp using current policy (not under added extra noise)
            # Recompute logp deterministically by querying the policy distribution for the sampled z (we used policy.sample_action earlier that returned logp when deterministic=False).
            # To avoid re-evaluating internals, we will recompute logp now:
            mean_t, log_std_t = self.policy.forward(torch.from_numpy(obs).unsqueeze(0).to(self.device))
            std_t = torch.exp(log_std_t).unsqueeze(0)
            # invert tanh scaling: we need the pre-tanh latent z for the action we applied
            # approximate by atanh((action - mid)/half_range)
            half_range = (self.policy.i_max - self.policy.i_min) / 2.0
            mid = (self.policy.i_max + self.policy.i_min) / 2.0
            # clip to avoid atanh domain error
            y = np.clip((action_np - mid) / half_range, -0.999999, 0.999999)
            z_applied = 0.5 * np.log((1 + y) / (1 - y))  # atanh(y)
            z_tensor = torch.from_numpy(z_applied.astype(np.float32)).unsqueeze(0).to(self.device)
            # compute log prob of z under gaussian
            var = (std_t ** 2)
            logp_tensor = -0.5 * (((z_tensor - mean_t) ** 2) / var + 2 * log_std_t + np.log(2 * np.pi))
            logp_scalar = logp_tensor.sum(dim=1)  # shape (1,)
            trajectory.append((obs.copy(), action_np.copy(), logp_scalar))

            # wait control interval (sim is running continuously in its own loop)
            time.sleep(self.dt)
            t += self.dt

            # termination condition: if recovered and we've held recovered for one control interval, we can stop early
            if recovered_at is not None:
                # optionally break immediately
                break

        # If never recovered set T = max_episode_time else T = recovered_at
        T = recovered_at if recovered_at is not None else self.max_episode_time
        # compute indicator I: 1 if T < recovery_time_threshold AND final tilt rates < rate_thresh
        final_phi_dot = last_obs[2]
        final_theta_dot = last_obs[3]
        I_indicator = 1 if (T < self.recovery_time_threshold and abs(final_phi_dot) < self.rate_thresh and abs(final_theta_dot) < self.rate_thresh) else 0
        V = V_time
        # final episodic reward
        total_reward = -(T * self.w_t + E * self.w_a + V * self.w_v) + self.B * I_indicator

        episodic_info = {"T": float(T), "E": float(E), "V": float(V), "I": int(I_indicator), "reward": float(total_reward), "steps": len(trajectory)}

        return episodic_info, trajectory

    def update_policy(self, trajectories):
        """
        trajectories: list of trajectories from run_episode; each trajectory is list of (obs, action, logp_tensor)
        We compute episodic returns (each episode has a scalar reward) and apply REINFORCE update.
        For simplicity, we assign the same episodic return to each timestep's logp (standard episodic REINFORCE).
        """
        # build batched training tensors
        all_logps = []
        all_returns = []
        for episodic_info, traj in trajectories:
            R = episodic_info["reward"]
            for (_, _, logp) in traj:
                all_logps.append(logp)
                all_returns.append(R)

        if len(all_logps) == 0:
            return None

        logp_tensor = torch.cat(all_logps).to(self.device)  # shape (N,)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=self.device)  # shape (N,)

        # normalize returns for stability
        returns_mean = returns_tensor.mean()
        returns_std = returns_tensor.std(unbiased=False) + 1e-8
        returns_norm = (returns_tensor - returns_mean) / returns_std

        # loss = - mean( logp * return )
        loss = - (logp_tensor * returns_norm).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return float(loss.item())

    def train(self, n_epochs=100, episodes_per_epoch=8, exploration_scale=1.0, save_every=10, model_path=None):
        """
        Train the policy with REINFORCE.
        """
        best_reward = -1e9
        for epoch in range(1, n_epochs + 1):
            epoch_trajs = []
            epoch_rewards = []
            for ep in range(episodes_per_epoch):
                # pick a random initial condition from CSV
                idx = np.random.randint(0, self.n_init)
                init_state = self.init_states[idx]
                episodic_info, traj = self.run_episode(init_state, train=True, exploration_std_scale=exploration_scale)
                epoch_trajs.append((episodic_info, traj))
                epoch_rewards.append(episodic_info["reward"])
                print(f"[Epoch {epoch}/{n_epochs}] Ep {ep+1}/{episodes_per_epoch}: reward={episodic_info['reward']:.3f}, T={episodic_info['T']:.3f}, E={episodic_info['E']:.3f}, V={episodic_info['V']:.3f}, I={episodic_info['I']}, steps={episodic_info['steps']}")

            # policy update
            loss = self.update_policy(epoch_trajs)
            avg_reward = float(np.mean(epoch_rewards))
            print(f"Epoch {epoch} update: loss={loss:.6f}, avg_reward={avg_reward:.4f}")

            # optional save
            if model_path and (epoch % save_every == 0):
                torch.save(self.policy.state_dict(), model_path)
                print(f"Saved policy to {model_path}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                # optionally save best
                if model_path:
                    torch.save(self.policy.state_dict(), model_path + ".best")
        return

    def evaluate(self, n_episodes=20, deterministic=True, render=False):
        stats = []
        for i in range(n_episodes):
            idx = np.random.randint(0, self.n_init)
            init_state = self.init_states[idx]
            episodic_info, _ = self.run_episode(init_state, train=False)
            stats.append(episodic_info)
            print(f"[Eval {i+1}/{n_episodes}] reward={episodic_info['reward']:.3f}, T={episodic_info['T']:.3f}, E={episodic_info['E']:.3f}, V={episodic_info['V']:.3f}, I={episodic_info['I']}")
        arr = pd.DataFrame(stats)
        print("Evaluation summary:")
        print(arr.describe())
        return arr

# -----------------------
# CLI and run
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="RL controller for platform simulator")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--csv", default="vibration_data.csv", help="path to vibration_data.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--dt", type=float, default=0.05, help="control interval (s)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eps_per_epoch", type=int, default=8)
    parser.add_argument("--model_path", default="policy.pth")
    parser.add_argument("--i_min", type=float, default=-2.0)
    parser.add_argument("--i_max", type=float, default=2.0)
    parser.add_argument("--w_t", type=float, default=1.0)
    parser.add_argument("--w_a", type=float, default=0.1)
    parser.add_argument("--w_v", type=float, default=0.5)
    parser.add_argument("--B", type=float, default=50.0)
    parser.add_argument("--max_time", type=float, default=8.0)
    parser.add_argument("--angle_thresh", type=float, default=0.01)
    parser.add_argument("--rate_thresh", type=float, default=0.05)
    parser.add_argument("--motion_rate_thresh", type=float, default=0.2)
    parser.add_argument("--recovery_time_thresh", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    sim_client = SimClient(host=args.host, port=args.port)
    agent = REINFORCE_Agent(
        sim_client,
        csv_path=args.csv,
        dt_control=args.dt,
        max_episode_time=args.max_time,
        angle_thresh=args.angle_thresh,
        rate_thresh=args.rate_thresh,
        motion_rate_thresh=args.motion_rate_thresh,
        w_t=args.w_t,
        w_a=args.w_a,
        w_v=args.w_v,
        B=args.B,
        recovery_time_threshold=args.recovery_time_thresh,
        lr=args.lr,
        gamma=0.99,
        device="cpu",
        i_min=args.i_min,
        i_max=args.i_max,
    )

    if args.mode == "train":
        print("Starting training...")
        agent.train(n_epochs=args.epochs, episodes_per_epoch=args.eps_per_epoch, exploration_scale=1.0, save_every=10, model_path=args.model_path)
        print("Training complete. Saved model to", args.model_path)
    else:
        # eval mode
        print("Evaluating policy...")
        # try to load model if exists
        try:
            agent.policy.load_state_dict(torch.load(args.model_path, map_location="cpu"))
            print("Loaded policy from", args.model_path)
        except Exception:
            print("No saved model found at", args.model_path, "- using random policy")
        agent.evaluate(n_episodes=20)

if __name__ == "__main__":
    main()
