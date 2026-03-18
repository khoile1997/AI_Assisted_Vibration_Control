#!/usr/bin/env python3
"""
generate_csv.py  – standalone script that generates vibration_training_data.csv
Uses the same physics as PlatformPhysics but avoids importing gymnasium/sb3.
Runs multiple episodes (seismic + harmonic) with mixed PD / random control.
"""
import numpy as np
import csv, os

# ── Physical constants (mirrors PlatformConfig) ──────────────────────────────
SIDE   = 0.050
L      = 0.025
MASS   = 0.500
IXX    = MASS * SIDE**2 / 6.0
IYY    = IXX
K      = 500.0
C_DAMP = 2.0
ACT_MAX= 0.010
DT     = 0.001
TAU_A  = 1.0 / (2 * np.pi * 50)
I_MAX  = 3.0
NOISE_IMU = np.deg2rad(0.05)
NOISE_I   = 0.02
NOISE_X   = 2e-5

CORNERS = np.array([[-L,+L],[+L,+L],[+L,-L],[-L,-L]], dtype=float)  # (4,2)

def current_from_shaft(s_cmd, s_actual):
    err = s_cmd - s_actual
    return float(np.clip(0.5 + 2.0*abs(err)/ACT_MAX, 0, I_MAX))

def rk4_step(state, s_cmd, dist_torque):
    """state = [θx,θy,ωx,ωy, s1,s2,s3,s4]"""
    def deriv(st):
        θx,θy,ωx,ωy = st[:4]
        s = st[4:]
        # actuator lag
        ds = (s_cmd - s) / TAU_A
        # corner heights from tilt
        z  = CORNERS[:,0]*np.sin(θy) - CORNERS[:,1]*np.sin(θx)
        dz = CORNERS[:,0]*ωy        - CORNERS[:,1]*ωx
        F  = K*(s - z) + C_DAMP*(ds - dz)
        Tx = np.sum(F * CORNERS[:,1]) + dist_torque[0]
        Ty =-np.sum(F * CORNERS[:,0]) + dist_torque[1]
        dωx = Tx / IXX
        dωy = Ty / IYY
        return np.array([ωx, ωy, dωx, dωy, *ds])
    k1 = deriv(state)
    k2 = deriv(state + 0.5*DT*k1)
    k3 = deriv(state + 0.5*DT*k2)
    k4 = deriv(state + DT*k3)
    return state + (DT/6)*(k1+2*k2+2*k3+k4)

def pd_action(θx,θy,ωx,ωy, kp=0.3, kd=0.05):
    # returns shaft commands [s1..s4]
    # push up corners on the low side
    err = np.array([
        -CORNERS[:,1]*θx + CORNERS[:,0]*θy
    ]).flatten()  # approximate height error per corner
    derr= np.array([
        -CORNERS[:,1]*ωx + CORNERS[:,0]*ωy
    ]).flatten()
    raw = ACT_MAX/2 + kp*err + kd*derr
    return np.clip(raw, 0, ACT_MAX)

def ar1_disturbance(n, rng, scale=0.005, rho=0.98):
    out = np.zeros((n,2))
    x = np.zeros(2)
    for i in range(n):
        x = rho*x + np.sqrt(1-rho**2)*rng.normal(0, scale, 2)
        out[i] = x
    return out

def harmonic_disturbance(n, rng, freqs=(2,5,8)):
    t = np.arange(n)*DT
    out = np.zeros((n,2))
    for f in freqs:
        A = rng.uniform(0.001, 0.004)
        ph= rng.uniform(0, 2*np.pi, 2)
        out[:,0] += A*np.sin(2*np.pi*f*t + ph[0])
        out[:,1] += A*np.sin(2*np.pi*f*t + ph[1])
    return out

# ── Generate data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
rows = []
EPISODES = 40
STEPS    = int(10.0 / DT)   # 10 s episodes at 1 ms

for ep in range(EPISODES):
    state = np.zeros(8)
    # Random initial tilt
    state[0] = rng.uniform(-0.10, 0.10)  # θx rad
    state[1] = rng.uniform(-0.10, 0.10)  # θy rad
    state[2] = rng.uniform(-0.05, 0.05)
    state[3] = rng.uniform(-0.05, 0.05)
    state[4:] = rng.uniform(0.002, 0.008, 4)

    if ep % 3 == 0:
        dist = ar1_disturbance(STEPS, rng, scale=0.004+rng.uniform(0,0.003))
    else:
        dist = harmonic_disturbance(STEPS, rng)

    use_random = (ep % 5 == 4)

    for k in range(STEPS):
        θx,θy,ωx,ωy = state[:4]
        s = state[4:]

        if use_random:
            s_cmd = rng.uniform(0, ACT_MAX, 4)
        else:
            s_cmd = pd_action(θx,θy,ωx,ωy)

        # Sensor readings with noise
        x_angle = float(np.degrees(θx) + rng.normal(0, np.degrees(NOISE_IMU)))
        y_angle = float(np.degrees(θy) + rng.normal(0, np.degrees(NOISE_IMU)))
        xdot    = float(np.degrees(ωx) + rng.normal(0, 0.5))
        ydot    = float(np.degrees(ωy) + rng.normal(0, 0.5))

        I = [current_from_shaft(s_cmd[i], s[i]) + rng.normal(0, NOISE_I) for i in range(4)]
        I = [float(np.clip(v, 0, I_MAX)) for v in I]

        # Corner displacements in mm
        z_corner = CORNERS[:,0]*np.sin(θy) - CORNERS[:,1]*np.sin(θx)
        x_mm = [(float(s[i]*1000 + rng.normal(0, NOISE_X*1000))) for i in range(4)]

        rows.append([x_angle, y_angle, xdot, ydot,
                     I[0], I[1], I[2], I[3],
                     x_mm[0], x_mm[1], x_mm[2], x_mm[3]])

        state = rk4_step(state, s_cmd, dist[k])

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_path = "/home/claude/vibration_control/data/vibration_training_data.csv"
header = ["x_angle","y_angle","xdot","ydot","I1","I2","I3","I4","x1","x2","x3","x4"]

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

total = len(rows)
print(f"Done – {total:,} rows written to {out_path}")
print(f"Episodes: {EPISODES}  |  Steps/episode: {STEPS}  |  Duration: 10 s each")
