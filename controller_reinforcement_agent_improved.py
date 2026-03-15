"""
Improved Reinforcement Learning Controller for Platform Stabilization System

This module implements a policy-gradient (REINFORCE) reinforcement learning agent
for controlling a 4-actuator platform stabilization system. The agent learns to
minimize vibration and recover platform stability by controlling motor currents.

System Architecture:
    - Platform: Rectangular with 4 corner-mounted electromagnetic actuators
    - Sensors: Central IMU measuring phi (roll), theta (pitch), and derivatives
    - Control: Current-based actuation with TCP/IP communication to simulator
    - Learning: Episodic REINFORCE with shaped reward function

Author: [Your Name]
Master's Thesis: Active Vibration Control Using Deep Reinforcement Learning
Date: February 2026
"""

import argparse
import socket
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass


def convert_sensor_data_to_state(row):
    """
    Convert MATLAB-generated sensor data to platform state.
    
    This function implements the EXACT INVERSE of MATLAB's compute_accel_from_corners.
    
    MATLAB format has columns: [ax, ay, az, motor1_diplacement[mm], motor2_diplacement[mm], 
                                 motor3_diplacement[mm], motor4_diplacement[mm]]
    Where:
        ax, ay, az: Accelerometer readings in m/s² (NOT mm/s²!)
        motor1-4_diplacement: Motor displacements in mm
    
    MATLAB platform geometry:
        L1 = 10 mm (lateral length, along X)
        L2 = 10 mm (depth, along Y)
        
        Corner layout:
        Motor 1: [0,  0,  x1]  (origin)
        Motor 2: [L1, 0,  x2]  (along X)
        Motor 3: [L1, L2, x3]  (diagonal)
        Motor 4: [0,  L2, x4]  (along Y)
    
    Returns:
        dict: {'phi': roll (rad), 'theta': pitch (rad), 
               'phi_dot': roll rate (rad/s), 'theta_dot': pitch rate (rad/s)}
    """
    # MATLAB platform dimensions (CRITICAL: must match MATLAB code!)
    L1 = 10.0  # mm (lateral length)
    L2 = 10.0  # mm (depth)
    
    # Extract motor displacements (in mm)
    # Note: MATLAB has typo "diplacement" instead of "displacement"
    x1 = float(row.get('motor1_diplacement[mm]', row.get('x1', 0)))
    x2 = float(row.get('motor2_diplacement[mm]', row.get('x2', 0)))
    x3 = float(row.get('motor3_diplacement[mm]', row.get('x3', 0)))
    x4 = float(row.get('motor4_diplacement[mm]', row.get('x4', 0)))
    
    # Extract accelerometer readings (in m/s² - already correct unit!)
    ax = float(row['ax'])
    ay = float(row['ay'])
    az = float(row['az'])
    
    # Reconstruct corner positions in MATLAB's coordinate system
    corners = np.array([
        [0,  0,  x1],  # Motor 1 at origin
        [L1, 0,  x2],  # Motor 2 along X
        [L1, L2, x3],  # Motor 3 at diagonal
        [0,  L2, x4]   # Motor 4 along Y
    ], dtype=np.float64)
    
    # Compute platform orientation from corners using rotation matrix method
    # (Same method as MATLAB's compute_accel_from_corners)
    
    # Extract corner points
    A, B, C, D = corners[0], corners[1], corners[2], corners[3]
    
    # Build in-plane axes
    v1 = B - A  # Edge A->B (x-axis direction)
    v2 = D - A  # Edge A->D (y-axis direction)
    
    x_axis = v1 / np.linalg.norm(v1)
    
    # Gram-Schmidt orthogonalization
    v2_orth = v2 - (np.dot(v2, x_axis) * x_axis)
    y_axis = v2_orth / np.linalg.norm(v2_orth)
    
    # Z-axis (normal to platform)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Rotation matrix: body to world
    R_b2w = np.column_stack([x_axis, y_axis, z_axis])
    
    # Extract Euler angles (Z-Y-X intrinsic convention)
    r31 = R_b2w[2, 0]
    r32 = R_b2w[2, 1]
    r33 = R_b2w[2, 2]
    r21 = R_b2w[1, 0]
    r11 = R_b2w[0, 0]
    
    # Pitch
    theta = np.arcsin(np.clip(-r31, -1.0, 1.0))
    cos_theta = np.cos(theta)
    
    # Roll and Yaw
    if abs(cos_theta) > 1e-6:
        phi = np.arctan2(r32, r33)    # Roll
        psi = np.arctan2(r21, r11)    # Yaw (not used)
    else:
        phi = 0.0  # Gimbal lock
        psi = np.arctan2(-R_b2w[0, 1], R_b2w[1, 1])
    
    # Estimate angular rates from accelerometer deviation
    # Compute what static accelerometer should read
    g = 9.81  # m/s²
    g_world = np.array([0, 0, -g])
    R_w2b = R_b2w.T
    a_body_static = R_w2b @ g_world
    
    # Deviation from static reading indicates motion
    a_measured = np.array([ax, ay, az])
    a_deviation = a_measured - a_body_static
    
    # Estimate rates from deviation (simplified)
    # Roll rate affects lateral (Y) acceleration
    # Pitch rate affects longitudinal (X) acceleration
    phi_dot = a_deviation[1] / (L1 / 2000.0)    # Convert mm to m
    theta_dot = a_deviation[0] / (L2 / 2000.0)
    
    # Limit to reasonable range
    phi_dot = np.clip(phi_dot, -0.5, 0.5)
    theta_dot = np.clip(theta_dot, -0.5, 0.5)
    
    return {
        'phi': float(phi),
        'theta': float(theta),
        'phi_dot': float(phi_dot),
        'theta_dot': float(theta_dot)
    }


class PlatformState:
    """Represents the complete state of the platform at a given time."""
    phi: float          # Roll angle (rad)
    theta: float        # Pitch angle (rad)
    z: float           # Vertical displacement (m)
    phi_dot: float     # Roll rate (rad/s)
    theta_dot: float   # Pitch rate (rad/s)
    z_dot: float       # Vertical velocity (m/s)
    currents: List[float]  # Motor currents [I1, I2, I3, I4] (A)
    timestamp: float   # Time (s)

    def to_observation(self) -> np.ndarray:
        """Convert to observation vector for RL agent."""
        return np.array([self.phi, self.theta, self.phi_dot, self.theta_dot], 
                       dtype=np.float32)

    @classmethod
    def from_sim_response(cls, response: dict, timestamp: float = 0.0):
        """Parse simulator response into PlatformState."""
        if isinstance(response, dict) and response.get("ok") and "data" in response:
            data = response["data"]
            state = data.get("state", {})
            currents = data.get("currents", [0.0] * 4)
        elif isinstance(response, dict) and "state" in response:
            state = response["state"]
            currents = response.get("currents", [0.0] * 4)
        else:
            state = response if isinstance(response, dict) else {}
            currents = state.get("currents", [0.0] * 4)
        
        return cls(
            phi=float(state.get("phi", 0.0)),
            theta=float(state.get("theta", 0.0)),
            z=float(state.get("z", 0.0)),
            phi_dot=float(state.get("phi_dot", 0.0)),
            theta_dot=float(state.get("theta_dot", 0.0)),
            z_dot=float(state.get("z_dot", 0.0)),
            currents=list(map(float, currents[:4])),
            timestamp=timestamp
        )


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    recovery_time: float      # Time to recovery (s), or max_time if not recovered
    actuation_effort: float   # Integrated sum of I_i^2 * dt
    vibration_time: float     # Time with excessive angular rates
    recovered: bool           # Whether platform recovered successfully
    reward: float            # Total episodic reward
    steps: int              # Number of control steps
    initial_state: np.ndarray  # Starting condition
    final_state: np.ndarray    # Ending condition

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/saving."""
        return {
            'T': self.recovery_time,
            'E': self.actuation_effort,
            'V': self.vibration_time,
            'I': int(self.recovered),
            'reward': self.reward,
            'steps': self.steps
        }


# ============================================================================
# SIMULATOR COMMUNICATION CLIENT
# ============================================================================

class SimulatorClient:
    """
    TCP/IP client for communicating with the platform simulator.
    
    Protocol: JSON messages over TCP, one command per connection.
    Commands: set_currents, get_state, set_state, set_limits
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5005, 
                 timeout: float = 2.0, max_retries: int = 5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify that simulator is reachable."""
        try:
            with socket.create_connection((self.host, self.port), timeout=2.0):
                logger.info(f"Successfully connected to simulator at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Cannot connect to simulator: {e}")
            raise ConnectionError(
                f"Simulator not reachable at {self.host}:{self.port}. "
                "Ensure real_time_platform_sim.py is running."
            )
    
    def _send_command(self, cmd_dict: dict, retry: int = 0) -> Optional[dict]:
        """
        Send JSON command to simulator and return parsed response.
        
        Args:
            cmd_dict: Command dictionary to send
            retry: Current retry attempt number
            
        Returns:
            Parsed JSON response or None on failure
        """
        message = json.dumps(cmd_dict) + "\n"
        
        try:
            with socket.create_connection((self.host, self.port), 
                                        timeout=self.timeout) as sock:
                sock.sendall(message.encode("utf-8"))
                sock.settimeout(self.timeout)
                
                # Read response
                response_bytes = b""
                while True:
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response_bytes += chunk
                    except socket.timeout:
                        break
                
                if not response_bytes:
                    logger.warning(f"Empty response for command: {cmd_dict.get('cmd')}")
                    if retry < self.max_retries:
                        time.sleep(0.2 * (retry + 1))
                        return self._send_command(cmd_dict, retry + 1)
                    return None
                
                # Parse JSON response
                response_text = response_bytes.decode("utf-8").strip()
                lines = [ln for ln in response_text.splitlines() if ln.strip()]
                parsed = [json.loads(ln) for ln in lines]
                return parsed[0] if parsed else None
                
        except socket.timeout:
            logger.warning(f"Socket timeout for command: {cmd_dict.get('cmd')}")
            if retry < self.max_retries:
                time.sleep(0.2 * (retry + 1))
                return self._send_command(cmd_dict, retry + 1)
            return None
            
        except Exception as e:
            logger.error(f"Communication error: {e}")
            if retry < self.max_retries:
                time.sleep(0.2 * (retry + 1))
                return self._send_command(cmd_dict, retry + 1)
            raise RuntimeError(f"Failed to communicate with simulator: {e}")
    
    def set_currents(self, currents: List[float]) -> Optional[dict]:
        """Set motor currents [I1, I2, I3, I4] in Amperes."""
        if len(currents) != 4:
            raise ValueError(f"Expected 4 currents, got {len(currents)}")
        return self._send_command({
            "cmd": "set_currents",
            "currents": list(map(float, currents))
        })
    
    def get_state(self) -> Optional[PlatformState]:
        """Get current platform state."""
        response = self._send_command({"cmd": "get_state"})
        if response is None:
            return None
        return PlatformState.from_sim_response(response, time.time())
    
    def set_state(self, phi: float = 0.0, theta: float = 0.0, 
                  phi_dot: float = 0.0, theta_dot: float = 0.0,
                  z: float = 0.0, z_dot: float = 0.0) -> Optional[dict]:
        """Set platform state (for episode initialization)."""
        return self._send_command({
            "cmd": "set_state",
            "state": {
                "phi": float(phi),
                "theta": float(theta),
                "phi_dot": float(phi_dot),
                "theta_dot": float(theta_dot),
                "z": float(z),
                "z_dot": float(z_dot)
            }
        })
    
    def set_limits(self, I_min: Optional[float] = None, 
                   I_max: Optional[float] = None,
                   F_max: Optional[float] = None) -> Optional[dict]:
        """Set actuator limits."""
        cmd = {"cmd": "set_limits"}
        if I_min is not None:
            cmd["I_min"] = float(I_min)
        if I_max is not None:
            cmd["I_max"] = float(I_max)
        if F_max is not None:
            cmd["F_max"] = float(F_max)
        return self._send_command(cmd)
    
    def reset_system(self):
        """Reset system to zero state and currents."""
        self.set_state(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.set_currents([0.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)  # Allow system to settle


# ============================================================================
# POLICY NETWORK
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Neural network policy for platform control.
    
    Architecture:
        - Input: [phi, theta, phi_dot, theta_dot] (4D observation)
        - Hidden: Configurable MLP with ReLU activations
        - Output: Gaussian distribution over 4 motor currents
        
    The policy outputs a mean and learned log-std for each actuator,
    then samples from a diagonal Gaussian and applies tanh squashing
    to bound outputs to [I_min, I_max].
    """
    
    def __init__(self, 
                 obs_dim: int = 4,
                 action_dim: int = 4,
                 hidden_sizes: Tuple[int, ...] = (128, 128),
                 activation: nn.Module = nn.ReLU,
                 init_log_std: float = -1.0,
                 i_min: float = -2.0,
                 i_max: float = 2.0):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.i_min = float(i_min)
        self.i_max = float(i_max)
        
        # Build MLP backbone
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation(),
            ])
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Policy head: outputs mean of action distribution
        self.mean_head = nn.Linear(prev_size, action_dim)
        
        # Learnable log standard deviation (diagonal covariance)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Small initialization for policy head
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            obs: Observation tensor, shape (batch, obs_dim) or (obs_dim,)
            
        Returns:
            mean: Mean of action distribution, shape (batch, action_dim)
            log_std: Log standard deviation, shape (action_dim,)
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)
        return mean, self.log_std
    
    def sample_action(self, obs: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, Optional[float], dict]:
        """
        Sample action from policy given observation.
        
        Args:
            obs: Observation array, shape (obs_dim,) or (batch, obs_dim)
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Sampled action (currents), shape (action_dim,)
            log_prob: Log probability of sampled action (None if deterministic)
            info: Dictionary with intermediate tensors for training
        """
        # Convert to tensor
        single = obs.ndim == 1
        obs_tensor = torch.from_numpy(obs.astype(np.float32))
        if single:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.forward(obs_tensor)
            std = torch.exp(log_std)
            
            if deterministic:
                # Use mean action (no exploration)
                z = mean
                log_prob = None
            else:
                # Sample from Gaussian
                eps = torch.randn_like(mean)
                z = mean + eps * std
                
                # Compute log probability of sampled action
                var = std.pow(2)
                log_prob = -0.5 * (((z - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
                log_prob = log_prob.sum(dim=-1)
            
            # Apply tanh squashing and scale to [I_min, I_max]
            z_squashed = torch.tanh(z)
            action = z_squashed * (self.i_max - self.i_min) / 2.0 + (self.i_max + self.i_min) / 2.0
        
        # Convert to numpy
        action_np = action[0].cpu().numpy() if single else action.cpu().numpy()
        log_prob_np = None if log_prob is None else (log_prob[0].item() if single else log_prob.cpu().numpy())
        
        info = {
            'mean': mean,
            'log_std': log_std,
            'z': z,
            'std': std
        }
        
        return action_np, log_prob_np, info
    
    def evaluate_actions(self, obs: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions.
        
        Used during training to compute policy gradients.
        
        Args:
            obs: Observation tensor, shape (batch, obs_dim)
            actions: Action tensor (currents), shape (batch, action_dim)
            
        Returns:
            log_probs: Log probabilities, shape (batch,)
            entropy: Policy entropy, shape (batch,)
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        # Invert tanh scaling to get pre-squashed latent
        # action = tanh(z) * scale + offset
        # z = atanh((action - offset) / scale)
        scale = (self.i_max - self.i_min) / 2.0
        offset = (self.i_max + self.i_min) / 2.0
        
        # Clamp to avoid atanh numerical issues
        normalized = torch.clamp((actions - offset) / scale, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + normalized) / (1 - normalized))  # atanh
        
        # Compute log probability of the pre-squash Gaussian
        var = std.pow(2)
        log_prob = -0.5 * (((z - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        # For on-policy REINFORCE we use the Gaussian log prob directly.
        # No tanh correction needed (that's for off-policy algorithms like SAC).
        # The policy gradient estimator only needs consistent probability ratios.

        # Compute entropy
        entropy = 0.5 * (torch.log(2 * np.pi * var) + 1).sum(dim=-1)
        
        return log_prob, entropy


# ============================================================================
# REINFORCEMENT LEARNING AGENT
# ============================================================================

class REINFORCEAgent:
    """
    REINFORCE policy gradient agent for platform stabilization.
    
    The agent learns to control 4 motor currents to minimize vibration
    and recover platform stability. Uses episodic rewards with shaped
    components: time to recovery, actuation effort, and vibration time.
    """
    
    def __init__(self,
                 simulator: SimulatorClient,
                 initial_conditions_path: str,
                 dt_control: float = 0.05,
                 max_episode_time: float = 8.0,
                 angle_threshold: float = 0.01,
                 rate_threshold: float = 0.05,
                 motion_rate_threshold: float = 0.2,
                 w_time: float = 1.0,
                 w_effort: float = 0.1,
                 w_vibration: float = 0.5,
                 bonus: float = 50.0,
                 recovery_time_bonus_threshold: float = 1.0,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 i_min: float = -2.0,
                 i_max: float = 2.0,
                 hidden_sizes: Tuple[int, ...] = (128, 128),
                 device: str = "cpu"):
        """
        Initialize REINFORCE agent.
        
        Args:
            simulator: SimulatorClient instance
            initial_conditions_path: Path to CSV with initial states
            dt_control: Control timestep (seconds)
            max_episode_time: Maximum episode duration (seconds)
            angle_threshold: Recovery threshold for angles (radians)
            rate_threshold: Recovery threshold for angular rates (rad/s)
            motion_rate_threshold: Threshold for excessive motion (rad/s)
            w_time: Weight for time-to-recovery penalty
            w_effort: Weight for actuation effort penalty
            w_vibration: Weight for vibration time penalty
            bonus: Bonus reward for fast recovery
            recovery_time_bonus_threshold: Time threshold for bonus (seconds)
            learning_rate: Optimizer learning rate
            gamma: Discount factor (not used in episodic setting)
            i_min: Minimum actuator current (A)
            i_max: Maximum actuator current (A)
            hidden_sizes: MLP hidden layer sizes
            device: Torch device ('cpu' or 'cuda')
        """
        self.sim = simulator
        self.dt = dt_control
        self.max_episode_time = max_episode_time
        
        # Recovery thresholds
        self.angle_threshold = angle_threshold
        self.rate_threshold = rate_threshold
        self.motion_rate_threshold = motion_rate_threshold
        
        # Reward weights
        self.w_time = w_time
        self.w_effort = w_effort
        self.w_vibration = w_vibration
        self.bonus = bonus
        self.recovery_time_bonus_threshold = recovery_time_bonus_threshold
        
        # Current limits
        self.i_min = i_min
        self.i_max = i_max
        
        # Training parameters
        self.gamma = gamma
        self.device = device
        
        # Load initial conditions
        self.initial_conditions = self._load_initial_conditions(initial_conditions_path)
        logger.info(f"Loaded {len(self.initial_conditions)} initial conditions")
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            obs_dim=4,
            action_dim=4,
            hidden_sizes=hidden_sizes,
            i_min=i_min,
            i_max=i_max
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training statistics
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=100)
        # Running return history for stable normalisation (last 200 episodes)
        self._return_history = deque(maxlen=200)
        # Separate history for failed episodes to compute stable baseline
        self._failed_return_history = deque(maxlen=200)
        
        logger.info(f"Initialized REINFORCE agent with policy: {self.policy}")
    
    def _load_initial_conditions(self, csv_path: str) -> np.ndarray:
        """
        Load initial conditions from CSV file.
        
        Supports two formats:
        1. Old format: columns [phi, theta, phi_dot, theta_dot]
        2. New format: columns [ax, ay, az, x1, x2, x3, x4]
        
        The new format is automatically converted to the old format.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Initial conditions file not found: {csv_path}\n"
                "Expected CSV with columns: phi, theta, phi_dot, theta_dot OR ax, ay, az, x1, x2, x3, x4"
            )
        
        # Case-insensitive column matching
        cols = {c.lower(): c for c in df.columns}
        
        # Check for old format (phi, theta, phi_dot, theta_dot)
        old_format_required = ["phi", "theta", "phi_dot", "theta_dot"]
        has_old_format = all(col in cols for col in old_format_required)
        
        # Check for new format (ax, ay, az, x1, x2, x3, x4)
        new_format_required = ["ax", "ay", "az", "x1", "x2", "x3", "x4"]
        has_new_format = all(col in cols for col in new_format_required)
        
        if has_old_format:
            # Old format - use directly
            logger.info(f"Loading {len(df)} initial conditions (old format: phi, theta, phi_dot, theta_dot)")
            states = df[[cols[c] for c in old_format_required]].to_numpy(dtype=np.float32)
            
        elif has_new_format:
            # New format - convert sensor data to platform states
            logger.info(f"Loading {len(df)} initial conditions (new format: ax, ay, az, x1-x4)")
            logger.info("Converting sensor data to platform states...")
            
            # Convert each row
            converted_states = []
            for _, row in df.iterrows():
                state_dict = convert_sensor_data_to_state(row)
                converted_states.append([
                    state_dict['phi'],
                    state_dict['theta'],
                    state_dict['phi_dot'],
                    state_dict['theta_dot']
                ])
            
            states = np.array(converted_states, dtype=np.float32)
            
            # Log sample conversion for verification
            if len(df) > 0:
                sample_row = df.iloc[0]
                sample_state = states[0]
                logger.info(f"Sample conversion verification:")
                logger.info(f"  Input:  x1={sample_row['x1']:.2f}mm, x2={sample_row['x2']:.2f}mm, "
                          f"x3={sample_row['x3']:.2f}mm, x4={sample_row['x4']:.2f}mm")
                logger.info(f"          ax={sample_row['ax']:.1f}mm/s², ay={sample_row['ay']:.1f}mm/s², "
                          f"az={sample_row['az']:.1f}mm/s²")
                logger.info(f"  Output: φ={sample_state[0]:.4f}rad, θ={sample_state[1]:.4f}rad, "
                          f"φ̇={sample_state[2]:.4f}rad/s, θ̇={sample_state[3]:.4f}rad/s")
                logger.info(f"Converted {len(states)} sensor readings successfully")
        else:
            raise ValueError(
                f"CSV format not recognized. Expected either:\n"
                f"  Old format: {old_format_required}\n"
                f"  New format: {new_format_required}\n"
                f"Found columns: {list(df.columns)}"
            )
        
        # Validate values
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            raise ValueError("Initial conditions contain NaN or Inf values after conversion")
        
        return states
    
    def _check_recovery(self, state: PlatformState) -> bool:
        """Check if platform has recovered (angles and rates below thresholds)."""
        angle_ok = (abs(state.phi) <= self.angle_threshold and 
                   abs(state.theta) <= self.angle_threshold)
        rate_ok = (abs(state.phi_dot) <= self.rate_threshold and 
                  abs(state.theta_dot) <= self.rate_threshold)
        return angle_ok and rate_ok
    
    def _compute_reward(self, metrics: EpisodeMetrics) -> float:
        """
        Compute shaped episodic reward.
        
        Reward = -(T*w_t + E*w_e + V*w_v) + B*I
        
        where:
            T: Time to recovery (or max_episode_time if not recovered)
            E: Actuation effort (integral of sum(I_i^2) dt)
            V: Time spent with excessive angular rates
            I: Binary indicator (1 if fast recovery, 0 otherwise)
            B: Bonus for fast recovery
        """
        T = metrics.recovery_time
        E = metrics.actuation_effort
        V = metrics.vibration_time
        I = 1.0 if (metrics.recovered and 
                   T < self.recovery_time_bonus_threshold) else 0.0
        
        reward = -(T * self.w_time + 
                  E * self.w_effort + 
                  V * self.w_vibration) + self.bonus * I
        
        return reward
    
    def run_episode(self, 
                   initial_state: np.ndarray,
                   training: bool = True,
                   exploration_noise: float = 0.0) -> Tuple[EpisodeMetrics, List]:
        """
        Run one episode of platform control.
        
        Args:
            initial_state: Initial [phi, theta, phi_dot, theta_dot]
            training: If True, collect trajectory for training
            exploration_noise: Additional Gaussian noise std for exploration
            
        Returns:
            metrics: EpisodeMetrics with performance statistics
            trajectory: List of (obs, action, log_prob) tuples
        """
        # Initialize platform state - retry until simulator acknowledges
        phi0, theta0, phi_dot0, theta_dot0 = initial_state
        for _attempt in range(10):
            resp = self.sim.set_state(phi0, theta0, phi_dot0, theta_dot0, 0.0, 0.0)
            if resp is not None:
                break
            logger.warning(f"set_state failed (attempt {_attempt+1}/10), retrying...")
            time.sleep(0.3)
        self.sim.set_currents([0.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)  # Allow state to be applied

        # Warm-up: wait until simulator responds with a valid state before
        # entering the control loop.  This handles the case where the sim is
        # momentarily busy after set_state.
        last_state = None
        for _warmup in range(20):
            last_state = self.sim.get_state()
            if last_state is not None:
                break
            logger.warning(f"Warm-up get_state failed (attempt {_warmup+1}/20), retrying...")
            time.sleep(0.2)

        if last_state is None:
            logger.error("Simulator did not respond after warm-up; skipping episode")
            # Return a zero-effort failed episode so training can continue
            dummy = EpisodeMetrics(
                recovery_time=self.max_episode_time,
                actuation_effort=0.0,
                vibration_time=0.0,
                recovered=False,
                reward=-self.max_episode_time * self.w_time,
                steps=0,
                initial_state=initial_state.copy(),
                final_state=initial_state.copy()
            )
            return dummy, []

        # Episode variables
        t = 0.0
        recovery_time = None
        actuation_effort = 0.0
        vibration_time = 0.0
        trajectory = []

        # Control loop
        while t < self.max_episode_time:
            # Get current state - on failure reuse last known state for this step
            state = self.sim.get_state()
            if state is None:
                logger.warning("get_state failed mid-episode; reusing last known state")
                state = last_state   # keep simulator loop alive
            else:
                last_state = state

            obs = state.to_observation()
            
            # Check if recovered
            if recovery_time is None and self._check_recovery(state):
                recovery_time = t
                # Continue episode briefly to ensure stability
            
            # Track vibration time
            max_rate = max(abs(state.phi_dot), abs(state.theta_dot))
            if max_rate > self.motion_rate_threshold:
                vibration_time += self.dt
            
            # Select action
            action, log_prob, _ = self.policy.sample_action(
                obs, deterministic=not training
            )
            
            # Add exploration noise if training
            if training and exploration_noise > 0:
                noise = np.random.normal(0, exploration_noise, size=action.shape)
                action = np.clip(action + noise, self.i_min, self.i_max)
            
            # Apply action - if it fails the sim keeps its last currents, log and move on
            if self.sim.set_currents(action.tolist()) is None:
                logger.warning("set_currents failed this step; simulator retains previous currents")
            
            # Accumulate actuation effort: E += sum(I_i^2) * dt
            actuation_effort += np.sum(action ** 2) * self.dt
            
            # Record trajectory for training
            if training:
                # Recompute log_prob with policy for the actual action (WITH gradients)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                action_tensor = torch.from_numpy(action).unsqueeze(0).to(self.device)
                # DO NOT use no_grad() - we need gradients for training!
                log_prob_tensor, _ = self.policy.evaluate_actions(
                    obs_tensor, action_tensor
                )
                trajectory.append((obs, action, log_prob_tensor))
            
            # Wait for control interval
            time.sleep(self.dt)
            t += self.dt
            
            # Early termination if recovered and stable
            if recovery_time is not None and (t - recovery_time) > 0.5:
                break
        
        # Compute final metrics
        T = recovery_time if recovery_time is not None else self.max_episode_time
        recovered = (recovery_time is not None and 
                    recovery_time < self.recovery_time_bonus_threshold)
        
        metrics = EpisodeMetrics(
            recovery_time=T,
            actuation_effort=actuation_effort,
            vibration_time=vibration_time,
            recovered=recovered,
            reward=0.0,  # Will be computed next
            steps=len(trajectory),
            initial_state=initial_state.copy(),
            final_state=last_state.to_observation() if last_state else initial_state
        )
        
        metrics.reward = self._compute_reward(metrics)
        
        self.episode_count += 1
        self.recent_rewards.append(metrics.reward)
        
        return metrics, trajectory
    
    def update_policy(self, batch_trajectories: List[Tuple[EpisodeMetrics, List]]):
        """
        Update policy using REINFORCE algorithm.
        
        Args:
            batch_trajectories: List of (metrics, trajectory) tuples
        """
        if not batch_trajectories:
            return None
        
        # Collect all log probabilities and returns
        log_probs = []
        returns = []
        
        for metrics, trajectory in batch_trajectories:
            R = metrics.reward
            
            # CRITICAL FIX: Use episode-level log prob, not timestep-level
            # REINFORCE should have ONE gradient per episode, not per timestep
            # Sum log probs across the trajectory (equivalent to product of probabilities)
            trajectory_log_probs = [log_prob for obs, action, log_prob in trajectory]
            if trajectory_log_probs:
                episode_log_prob = torch.stack(trajectory_log_probs).sum()
                log_probs.append(episode_log_prob)
                returns.append(R)
        
        if not log_probs:
            return None
        
        # Convert to tensors
        log_prob_tensor = torch.cat(log_probs).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns using a RUNNING baseline rather than within-batch std.
        # When all episodes fail with nearly identical rewards the within-batch std
        # collapses to ~0, turning the normalised returns into pure noise and making
        # the gradient useless.  A running mean/std over recent history gives a
        # stable, meaningful signal even early in training.
        # Separate tracking: failed episodes vs. all episodes
        # This prevents successful episodes from corrupting the baseline
        self._return_history.extend(returns)
        
        # Track failed episodes separately (returns < 0 means failed)
        failed_returns = [r for r in returns if r < 0]
        if failed_returns:
            self._failed_return_history.extend(failed_returns)
        
        # Use ONLY failed episodes for baseline to avoid instability
        # When successes occur (R >> 0), using them in baseline makes
        # subsequent failures look anomalously bad, causing negative loss
        if len(self._failed_return_history) >= 10:
            baseline = float(np.mean(self._failed_return_history))
        else:
            # Fallback: use all returns if not enough failures yet
            baseline = float(np.mean(self._return_history)) if len(self._return_history) >= 5 else 0.0
        
        returns_centered = returns_tensor - baseline
        # Policy gradient loss: -E[log π(a|s) * (R - baseline)]
        # This is the standard REINFORCE loss with baseline subtraction
        loss = -(log_prob_tensor * returns_centered).mean()
        # Policy gradient loss: -E[log π(a|s) * (R - baseline)]
        # This is the standard REINFORCE loss with baseline subtraction
        loss = -(log_prob_tensor * returns_centered).mean()

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self,
              n_epochs: int = 100,
              episodes_per_epoch: int = 8,
              exploration_noise: float = 0.05,
              save_every: int = 10,
              model_path: Optional[str] = None):
        """
        Train the policy using REINFORCE.
        
        Args:
            n_epochs: Number of training epochs
            episodes_per_epoch: Episodes per epoch (batch size)
            exploration_noise: Std of Gaussian exploration noise
            save_every: Save model every N epochs
            model_path: Path to save model checkpoints
        """
        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Episodes per epoch: {episodes_per_epoch}")
        logger.info(f"Exploration noise: {exploration_noise}")
        
        best_reward = float('-inf')
        
        for epoch in range(1, n_epochs + 1):
            epoch_trajectories = []
            epoch_rewards = []
            
            # Collect batch of episodes
            for ep in range(episodes_per_epoch):
                # Sample random initial condition
                idx = np.random.randint(0, len(self.initial_conditions))
                init_state = self.initial_conditions[idx]
                
                # Run episode
                metrics, trajectory = self.run_episode(
                    init_state,
                    training=True,
                    exploration_noise=exploration_noise
                )
                
                epoch_trajectories.append((metrics, trajectory))
                epoch_rewards.append(metrics.reward)
                
                logger.info(
                    f"[Epoch {epoch}/{n_epochs}] Episode {ep+1}/{episodes_per_epoch}: "
                    f"R={metrics.reward:.2f}, T={metrics.recovery_time:.2f}s, "
                    f"E={metrics.actuation_effort:.2f}, V={metrics.vibration_time:.2f}s, "
                    f"Recovered={metrics.recovered}"
                )
            
            # Update policy
            loss = self.update_policy(epoch_trajectories)
            avg_reward = np.mean(epoch_rewards)
            std_reward = np.std(epoch_rewards)

            if loss is None:
                logger.warning(
                    f"Epoch {epoch} Summary: "
                    f"Loss=N/A (no valid trajectories this epoch), "
                    f"Reward={avg_reward:.2f}±{std_reward:.2f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch} Summary: "
                    f"Loss={loss:.6f}, Reward={avg_reward:.2f}±{std_reward:.2f}"
                )
            
            # Save checkpoints
            if model_path and (epoch % save_every == 0 or epoch == n_epochs):
                self.save_model(model_path)
                logger.info(f"Saved checkpoint to {model_path}")
            
            # Save best model
            if model_path and avg_reward > best_reward:
                best_reward = avg_reward
                best_path = str(Path(model_path).with_suffix('.best.pth'))
                self.save_model(best_path)
                logger.info(f"New best model! Reward={best_reward:.2f}")
        
        logger.info("Training complete!")
    
    def evaluate(self, n_episodes: int = 20) -> pd.DataFrame:
        """
        Evaluate policy performance.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            DataFrame with episode statistics
        """
        logger.info(f"Evaluating policy over {n_episodes} episodes")
        
        results = []
        
        for i in range(n_episodes):
            # Sample random initial condition
            idx = np.random.randint(0, len(self.initial_conditions))
            init_state = self.initial_conditions[idx]
            
            # Run episode deterministically
            metrics, _ = self.run_episode(init_state, training=False)
            
            results.append(metrics.to_dict())
            
            logger.info(
                f"[Eval {i+1}/{n_episodes}] "
                f"R={metrics.reward:.2f}, T={metrics.recovery_time:.2f}s, "
                f"Recovered={metrics.recovered}"
            )
        
        df = pd.DataFrame(results)
        
        logger.info("\nEvaluation Summary:")
        logger.info(f"\n{df.describe()}")
        logger.info(f"\nRecovery Rate: {df['I'].mean() * 100:.1f}%")
        
        return df
    
    def save_model(self, path: str):
        """Save policy network state dict."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
        }, path)
    
    def load_model(self, path: str):
        """Load policy network state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        logger.info(f"Loaded model from {path} (episode {self.episode_count})")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for training/evaluation."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Controller for Platform Stabilization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
        help="Operation mode"
    )
    
    # Simulator connection
    parser.add_argument("--host", default="127.0.0.1", help="Simulator host")
    parser.add_argument("--port", type=int, default=5005, help="Simulator port")
    
    # Data
    parser.add_argument(
        "--csv", default="vibration_data.csv",
        help="Path to initial conditions CSV"
    )
    
    # Control parameters
    parser.add_argument(
        "--dt", type=float, default=0.05,
        help="Control timestep (seconds)"
    )
    parser.add_argument(
        "--max_time", type=float, default=8.0,
        help="Maximum episode duration (seconds)"
    )
    
    # Current limits
    parser.add_argument("--i_min", type=float, default=-2.0, help="Min current (A)")
    parser.add_argument("--i_max", type=float, default=2.0, help="Max current (A)")
    
    # Thresholds
    parser.add_argument(
        "--angle_thresh", type=float, default=0.01,
        help="Recovery angle threshold (rad)"
    )
    parser.add_argument(
        "--rate_thresh", type=float, default=0.05,
        help="Recovery rate threshold (rad/s)"
    )
    parser.add_argument(
        "--motion_rate_thresh", type=float, default=0.2,
        help="Excessive motion threshold (rad/s)"
    )
    parser.add_argument(
        "--recovery_time_thresh", type=float, default=1.0,
        help="Fast recovery time threshold for bonus (s)"
    )
    
    # Reward weights
    parser.add_argument("--w_time", type=float, default=1.0, help="Time weight")
    parser.add_argument("--w_effort", type=float, default=0.1, help="Effort weight")
    parser.add_argument("--w_vibration", type=float, default=0.5, help="Vibration weight")
    parser.add_argument("--bonus", type=float, default=50.0, help="Recovery bonus")
    
    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument(
        "--eps_per_epoch", type=int, default=8,
        help="Episodes per epoch"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden", nargs="+", type=int, default=[128, 128],
        help="Hidden layer sizes"
    )
    parser.add_argument(
        "--exploration_noise", type=float, default=0.05,
        help="Exploration noise std"
    )
    
    # Model I/O
    parser.add_argument(
        "--model_path", default="policy.pth",
        help="Path for saving/loading model"
    )
    parser.add_argument(
        "--load_model", action="store_true",
        help="Load existing model before training/eval"
    )
    
    # Evaluation
    parser.add_argument(
        "--n_eval", type=int, default=20,
        help="Number of evaluation episodes"
    )
    
    args = parser.parse_args()
    
    # Initialize simulator client
    logger.info("Connecting to simulator...")
    sim = SimulatorClient(host=args.host, port=args.port)
    
    # Initialize agent
    logger.info("Initializing agent...")
    agent = REINFORCEAgent(
        simulator=sim,
        initial_conditions_path=args.csv,
        dt_control=args.dt,
        max_episode_time=args.max_time,
        angle_threshold=args.angle_thresh,
        rate_threshold=args.rate_thresh,
        motion_rate_threshold=args.motion_rate_thresh,
        w_time=args.w_time,
        w_effort=args.w_effort,
        w_vibration=args.w_vibration,
        bonus=args.bonus,
        recovery_time_bonus_threshold=args.recovery_time_thresh,
        learning_rate=args.lr,
        i_min=args.i_min,
        i_max=args.i_max,
        hidden_sizes=tuple(args.hidden),
    )
    
    # Load existing model if requested
    if args.load_model:
        try:
            agent.load_model(args.model_path)
        except FileNotFoundError:
            logger.warning(f"Model file {args.model_path} not found, using random init")
    
    # Execute mode
    if args.mode == "train":
        agent.train(
            n_epochs=args.epochs,
            episodes_per_epoch=args.eps_per_epoch,
            exploration_noise=args.exploration_noise,
            save_every=10,
            model_path=args.model_path
        )
    else:  # eval
        results_df = agent.evaluate(n_episodes=args.n_eval)
        
        # Save results
        results_path = "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
