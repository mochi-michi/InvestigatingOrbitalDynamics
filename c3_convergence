#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt


# Physics helpers
GM_SUN = (2.0 * math.pi) ** 2  # 4 pi^2

def kepler_period(a: float) -> float:
    return a ** 1.5

def perihelion_speed(a: float, e: float) -> float:
    r_p = a * (1.0 - e)
    return math.sqrt(GM_SUN * (2.0 / r_p - 1.0 / a))

def init_perihelion(a: float, e: float) -> np.ndarray:
    x0 = a * (1.0 - e)
    y0 = 0.0
    v0 = perihelion_speed(a, e)
    vx0 = 0.0
    vy0 = v0
    return np.array([x0, y0, vx0, vy0], float)

def accel_newtonian(x: float, y: float) -> tuple[float, float]:
    r = math.hypot(x, y)
    r3 = r ** 3
    ax = -GM_SUN * x / r3
    ay = -GM_SUN * y / r3
    return ax, ay

# Integrators
def step_euler(y: np.ndarray, dt: float) -> np.ndarray:
    x, y_, vx, vy = y
    ax, ay = accel_newtonian(x, y_)
    x_new = x + dt * vx
    y_new = y_ + dt * vy
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay
    return np.array([x_new, y_new, vx_new, vy_new], float)

def step_leapfrog(y: np.ndarray, dt: float) -> np.ndarray:
    x, y_, vx, vy = y
    ax0, ay0 = accel_newtonian(x, y_)
    vx_half = vx + 0.5 * dt * ax0
    vy_half = vy + 0.5 * dt * ay0
    x_new = x + dt * vx_half
    y_new = y_ + dt * vy_half
    ax1, ay1 = accel_newtonian(x_new, y_new)
    vx_new = vx_half + 0.5 * dt * ax1
    vy_new = vy_half + 0.5 * dt * ay1
    return np.array([x_new, y_new, vx_new, vy_new], float)

def step_rk2(y: np.ndarray, dt: float) -> np.ndarray:
    x, y_, vx, vy = y
    ax1, ay1 = accel_newtonian(x, y_)
    x_mid = x + 0.5 * dt * vx
    y_mid = y_ + 0.5 * dt * vy
    vx_mid = vx + 0.5 * dt * ax1
    vy_mid = vy + 0.5 * dt * ay1
    ax2, ay2 = accel_newtonian(x_mid, y_mid)
    x_new = x + dt * vx_mid
    y_new = y_ + dt * vy_mid
    vx_new = vx + dt * ax2
    vy_new = vy + dt * ay2
    return np.array([x_new, y_new, vx_new, vy_new], float)

def step_rk4(y: np.ndarray, dt: float) -> np.ndarray:
    def deriv(state: np.ndarray) -> np.ndarray:
        x, y_, vx, vy = state
        ax, ay = accel_newtonian(x, y_)
        return np.array([vx, vy, ax, ay], float)
    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt * k1)
    k3 = deriv(y + 0.5 * dt * k2)
    k4 = deriv(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

STEP = {
    'euler': step_euler,
    'leapfrog': step_leapfrog,
    'rk2': step_rk2,
    'rk4': step_rk4,
}

# simulate function
def simulate(y0: np.ndarray, dt: float, n_steps: int, method: str):
    step = STEP[method]
    t = np.zeros(n_steps + 1)
    out = np.zeros((n_steps + 1, 4))
    y = y0.copy()
    t[0] = 0.0
    out[0] = y
    for i in range(1, n_steps + 1):
        y = step(y, dt)
        t[i] = i * dt
        out[i] = y
    return t, out

def main():
    parser = argparse.ArgumentParser(description='Standalone convergence test')
    parser.add_argument('--method', type=str, default='euler',
                        choices=['euler', 'leapfrog', 'rk2', 'rk4'])
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--e', type=float, default=0.0)
    parser.add_argument('--dt-min', type=float, default=1e-4)
    parser.add_argument('--dt-max', type=float, default=1e-2)
    parser.add_argument('--num', type=int, default=8)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    y0 = init_perihelion(args.a, args.e)
    x_exact = y0[0]
    T = kepler_period(args.a)

    dts = np.logspace(math.log10(args.dt_min), math.log10(args.dt_max), args.num)
    errors = []
    for dt in dts:
        n_steps = int(round(T / dt))
        _, out = simulate(y0, dt, n_steps, method=args.method)
        x_num = out[-1, 0]
        errors.append(abs(x_num - x_exact))
    dts = np.array(dts)
    errors = np.array(errors)

    log_dt = np.log10(dts)
    log_err = np.log10(errors)
    p, logA = np.polyfit(log_dt, log_err, 1)
    A = 10**logA
    print(f"Fitted order p ≈ {p:.3f} for method '{args.method}'")
    print(f"Fitted prefactor A ≈ {A:.3e}")

    if not args.no_plot:
        os.makedirs('outputs/figs', exist_ok=True)
        plt.figure()
        plt.loglog(dts, errors, 'o-', label='measured error')
        dt_fit = np.array([dts.min(), dts.max()])
        err_fit = A * dt_fit**p
        plt.loglog(dt_fit, err_fit, '--', label=f'fit: error ≈ {A:.2e} dt^{p:.2f}')
        plt.xlabel('dt')
        plt.ylabel('global error |x(T) - x_exact|')
        plt.title(f'Convergence test – {args.method}, order ≈ {p:.2f}')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fname = f'outputs/figs/c3_convergence_standalone_{args.method}.png'
        plt.savefig(fname, dpi=180)
        print(f'Saved convergence plot to {fname}')
        plt.show()

if __name__ == '__main__':
    main()
