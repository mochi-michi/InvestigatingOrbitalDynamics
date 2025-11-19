import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


method = "rk2"       # 'euler', 'leapfrog', 'rk2', 'rk4'
a = 0.387           # semi-major axis
e = 0.2056          # eccentricity 
dt = 1e-3           # step size
orbits = 1.0        # number of orbits
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
    ax0, ay0 = accel_newtonian(x, y_)
    vx_half = vx + 0.5 * dt_val * ax0
    vy_half = vy + 0.5 * dt_val * ay0
    x_new = x + dt_val * vx_half
    y_new = y_ + dt_val * vy_half
    ax1, ay1 = accel_newtonian(x_new, y_new)
    vx_new = vx_half + 0.5 * dt_val * ax1
    vy_new = vy_half + 0.5 * dt_val * ay1
    return np.array([x_new, y_new, vx_new, vy_new], float)


def step_rk2(y, dt_val):
    x, y_, vx, vy = y
    ax1, ay1 = accel_newtonian(x, y_)
    x_mid = x + 0.5 * dt_val * vx
    y_mid = y_ + 0.5 * dt_val * vy
    vx_mid = vx + 0.5 * dt_val * ax1
    vy_mid = vy + 0.5 * dt_val * ay1
    ax2, ay2 = accel_newtonian(x_mid, y_mid)
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


def simulate_fixed(y0, dt_val, n_steps, method_name):
    step = STEP[method_name]
    t = np.linspace(0.0, n_steps * dt_val, n_steps + 1)
    out = np.zeros((n_steps + 1, 4), float)
    y = y0.copy()
    out[0] = y
    for i in range(1, n_steps + 1):
        y = step(y, dt_val)
        out[i] = y
    return t, out



def rhs_scipy(t, y):
    x, y_, vx, vy = y
    ax, ay = accel_newtonian(x, y_)
    return np.array([vx, vy, ax, ay], float)



y0 = init_perihelion(a, e)
T = kepler_period(a)
t_end = orbits * T
n_steps = int(round(t_end / dt))

t_custom, out_custom = simulate_fixed(y0, dt, n_steps, method)

t_span = (float(t_custom[0]), float(t_custom[-1]))

sol = solve_ivp(
    rhs_scipy,
    t_span,
    y0,
    t_eval=t_custom,
    method="RK45",
)
y_scipy = sol.y.T 


diff = out_custom - y_scipy
rmse = np.sqrt(np.mean(diff**2, axis=0))

print(f"Method compared: {method}")
print("RMSE vs solve_ivp over [x, y, vx, vy]:", rmse)

if make_plot:
    plt.figure()
    plt.plot(out_custom[:, 0], out_custom[:, 1], label=f"{method}")
    plt.plot(y_scipy[:, 0], y_scipy[:, 1], "--", label="SciPy solve_ivp (RK45)")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title(f"C8: our {method} vs SciPy solve_ivp")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
