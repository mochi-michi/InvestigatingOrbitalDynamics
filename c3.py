import math
import numpy as np
import matplotlib.pyplot as plt

method = "rk4"         # "euler", "leapfrog", "rk2", "rk4"
a = 1.0
e = 0.0
dt_min = 1e-4
dt_max = 1e-2
num_dt = 8
make_plot = True

# Physics

GM_SUN = (2 * math.pi)**2

def kepler_period(a):
    return a**1.5

def perihelion_speed(a, e):
    r_p = a*(1-e)
    return math.sqrt(GM_SUN*(2/r_p - 1/a))

def init_perihelion(a, e):
    x0 = a*(1-e)
    y0 = 0
    v0 = perihelion_speed(a, e)
    return np.array([x0, y0, 0.0, v0], float)

def accel_newtonian(x, y):
    r = math.hypot(x, y)
    ax = -GM_SUN * x / r**3
    ay = -GM_SUN * y / r**3
    return ax, ay

# Integrators

def step_euler(y, dt):
    x, y_, vx, vy = y
    ax, ay = accel_newtonian(x, y_)
    return np.array([x+dt*vx, y_+dt*vy, vx+dt*ax, vy+dt*ay])

def step_leapfrog(y, dt):
    x, y_, vx, vy = y
    ax0, ay0 = accel_newtonian(x, y_)
    vxh = vx + 0.5*dt*ax0
    vyh = vy + 0.5*dt*ay0
    x_new = x + dt*vxh
    y_new = y_ + dt*vyh
    ax1, ay1 = accel_newtonian(x_new, y_new)
    return np.array([x_new, y_new, vxh+0.5*dt*ax1, vyh+0.5*dt*ay1])

def step_rk2(y, dt):
    x, y_, vx, vy = y
    ax1, ay1 = accel_newtonian(x, y_)
    xm = x + 0.5*dt*vx
    ym = y_ + 0.5*dt*vy
    vxm = vx + 0.5*dt*ax1
    vym = vy + 0.5*dt*ay1
    ax2, ay2 = accel_newtonian(xm, ym)
    return np.array([x+dt*vxm, y_+dt*vym, vx+dt*ax2, vy+dt*ay2])

def step_rk4(y, dt):
    def deriv(s):
        x, y_, vx, vy = s
        ax, ay = accel_newtonian(x, y_)
        return np.array([vx, vy, ax, ay])
    k1 = deriv(y)
    k2 = deriv(y + 0.5*dt*k1)
    k3 = deriv(y + 0.5*dt*k2)
    k4 = deriv(y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

STEP = {
    "euler": step_euler,
    "leapfrog": step_leapfrog,
    "rk2": step_rk2,
    "rk4": step_rk4
}

# Simulator

def simulate(y0, dt, n_steps, method):
    step = STEP[method]
    y = y0.copy()
    for _ in range(n_steps):
        y = step(y, dt)
    return y

# Run Test

dts = np.logspace(math.log10(dt_min), math.log10(dt_max), num_dt)
T = kepler_period(a)
y0 = init_perihelion(a, e)
x_exact = y0[0]

errors = []
for dt in dts:
    n_steps = int(round(T/dt))
    y_final = simulate(y0, dt, n_steps, method)
    errors.append(abs(y_final[0] - x_exact))

errors = np.array(errors)
p, logA = np.polyfit(np.log10(dts), np.log10(errors), 1)
A = 10**logA

print(f"Method: {method}")
print(f"Estimated order p ≈ {p:.3f}")
print(f"A ≈ {A:.3e}")

if make_plot:
    plt.figure()
    plt.loglog(dts, errors, "o-", label="Measured")
    fit_dt = np.array([dts.min(), dts.max()])
    plt.loglog(fit_dt, A*fit_dt**p, "--", label=f"Fit dt^{p:.2f}")
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title(f"Convergence ({method})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()
