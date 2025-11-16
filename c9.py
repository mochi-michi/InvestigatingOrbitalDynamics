import math
import numpy as np
import matplotlib.pyplot as plt

methods = ["euler", "leapfrog", "rk2", "rk4"]

# Mercury-like orbit (you can change these)
a = 0.387           # semi-major axis [AU]
e = 0.2056          # eccentricity

# I've set these so that all methods can be distinguished on the plot

dt = 0.1           # time step
orbits = 100.0       # number of orbits

make_plot = True 

GM_SUN = (2.0 * math.pi) ** 2


def kepler_period(a_val):
    return a_val ** 1.5


def perihelion_speed(a_val, e_val):
    r_p = a_val * (1.0 - e_val)
    return math.sqrt(GM_SUN * (2.0 / r_p - 1.0 / a_val))


def init_perihelion(a_val, e_val):
    x0 = a_val * (1.0 - e_val)
    y0 = 0.0
    v0 = perihelion_speed(a_val, e_val)
    vx0 = 0.0
    vy0 = v0
    return np.array([x0, y0, vx0, vy0], float)


def accel_newtonian(x, y):
    r = math.hypot(x, y)
    r3 = r ** 3
    ax = -GM_SUN * x / r3
    ay = -GM_SUN * y / r3
    return ax, ay


def energies(x, y, vx, vy):
    r = np.sqrt(x * x + y * y)
    v2 = vx * vx + vy * vy
    K = 0.5 * v2
    U = -GM_SUN / r
    E = K + U
    return K, U, E


# Integrators

def step_euler(y, dt_val):
    x, y_, vx, vy = y
    ax, ay = accel_newtonian(x, y_)
    return np.array(
        [
            x + dt_val * vx,
            y_ + dt_val * vy,
            vx + dt_val * ax,
            vy + dt_val * ay,
        ],
        float,
    )


def step_leapfrog(y, dt_val):
    x, y_, vx, vy = y

    # Half-kick
    ax0, ay0 = accel_newtonian(x, y_)
    vx_half = vx + 0.5 * dt_val * ax0
    vy_half = vy + 0.5 * dt_val * ay0

    # Drift
    x_new = x + dt_val * vx_half
    y_new = y_ + dt_val * vy_half

    # Second half-kick
    ax1, ay1 = accel_newtonian(x_new, y_new)
    vx_new = vx_half + 0.5 * dt_val * ax1
    vy_new = vy_half + 0.5 * dt_val * ay1

    return np.array([x_new, y_new, vx_new, vy_new], float)


def step_rk2(y, dt_val):
    x, y_, vx, vy = y
    ax1, ay1 = accel_newtonian(x, y_)

    # Midpoint predictor
    x_mid = x + 0.5 * dt_val * vx
    y_mid = y_ + 0.5 * dt_val * vy
    vx_mid = vx + 0.5 * dt_val * ax1
    vy_mid = vy + 0.5 * dt_val * ay1

    ax2, ay2 = accel_newtonian(x_mid, y_mid)

    # Corrector
    x_new = x + dt_val * vx_mid
    y_new = y_ + dt_val * vy_mid
    vx_new = vx + dt_val * ax2
    vy_new = vy + dt_val * ay2

    return np.array([x_new, y_new, vx_new, vy_new], float)


def step_rk4(y, dt_val):

    def deriv(state):
        x, y_, vx, vy = state
        ax, ay = accel_newtonian(x, y_)
        return np.array([vx, vy, ax, ay], float)

    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt_val * k1)
    k3 = deriv(y + 0.5 * dt_val * k2)
    k4 = deriv(y + dt_val * k3)

    return y + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


STEP = {
    "euler": step_euler,
    "leapfrog": step_leapfrog,
    "rk2": step_rk2,
    "rk4": step_rk4,
}



def simulate_orbit(y0: np.ndarray, dt_val: float, n_steps: int, method_name: str):
    step = STEP[method_name]
    t = np.linspace(0.0, n_steps * dt_val, n_steps + 1)
    out = np.zeros((n_steps + 1, 4), float)
    y = y0.copy()
    out[0] = y

    for i in range(1, n_steps + 1):
        y = step(y, dt_val)
        out[i] = y

    return t, out


T = kepler_period(a)
t_end = orbits * T
n_steps = int(round(t_end / dt))

y0 = init_perihelion(a, e)

energy_results = {}

print("Energy drift summary:")
for m in methods:
    t, out = simulate_orbit(y0, dt, n_steps, m)

    x = out[:, 0]
    y = out[:, 1]
    vx = out[:, 2]
    vy = out[:, 3]

    _, _, E = energies(x, y, vx, vy)
    energy_results[m] = (t, E)

    # Absolute and relative drift over the run
    drift = E[-1] - E[0]
    rel_drift = drift / abs(E[0])
    print(f"  {m:8s}: ΔE = {drift:+.3e},  relative drift = {rel_drift:+.3e}")



if make_plot:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6),
        constrained_layout=True
    )

  # Absolute energy 
    for m, (t, E) in energy_results.items():
        ax1.plot(t, E, label=m)
    ax1.set_ylabel("E(t) [AU^2 / yr^2]")
    ax1.set_title("C9: Energy behaviour for different integrators")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

  # Energy drift
    for m, (t, E) in energy_results.items():
        dE = E - E[0]
        ax2.plot(t, dE, label=m)
    ax2.set_xlabel("t [yr]")
    ax2.set_ylabel("ΔE(t) = E(t) - E(0)")
    ax2.grid(True, alpha=0.3)

    plt.show()
