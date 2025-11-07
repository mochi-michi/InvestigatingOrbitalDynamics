import math
import numpy as np
import matplotlib.pyplot as plt

# Constant
gm_sun = 4 * math.pi**2 
a = 0.387        # AU
e = 0.2056       # eccentricity
dt = 1e-4        # timestep
n_orbits = 2.0   # how many orbits to simulate

# Accelaration function

def acceleration(x, y, GM = gm_sun):
    r = math.sqrt(x**2 + y**2)
    return -GM * x/ r**3, -GM * y/ r**3

# Perihelion setup
def perihelion_speed(a, e, GM = gm_sun):
    return math.sqrt(GM * (1 + e) / (a * (1 - e)))

def perihelion(a, e, use_given_speed=False):
    # a is the axis, e is the eccentricity
    x0 = a * (1 - e) # radius
    y0 = 0.0
    vx0 = 0
    vy0 = perihelion_speed(a,e)
    return np.array([x0, y0, vx0, vy0])

def step_euler(y, dt):
    x, y_pos, vx, vy = y
    ax, ay = acceleration(x, y_pos)
    x_new = x + vx * dt
    y_new = y_pos + vy * dt
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    return np.array([x_new, y_new, vx_new, vy_new])

y = perihelion(a, e)
T = a**1.5
n_steps = int(n_orbits * T / dt)

t = np.empty(n_steps + 1)
out = np.empty((n_steps + 1, 4))
out[0] = y
t[0] = 0.0

for i in range(1, n_steps + 1):
    y = step_euler(y, dt)
    out[i] = y
    t[i] = i * dt


vx, vy = out[:, 2], out[:, 3]
K = 0.5 * (vx*vx + vy*vy)
U = -gm_sun / np.hypot(out[:,0], out[:,1])
E = K + U
print("Energy drift ΔE =", E[-1] - E[0])

yvals = out[:, 1]
vyvals = out[:, 3]
idx = np.where((yvals[:-1] <= 0) & (yvals[1:] > 0) & (vyvals[1:] > 0))[0][0]
print(f"Estimated period ≈ {t[idx+1]:.4f} yr (Kepler {T:.4f} yr)")

# plot
plt.figure()
plt.plot(out[:,0], out[:,1])
plt.plot([0], [0], "o")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x [AU]"); plt.ylabel("y [AU]")
plt.title(f"Euler orbit — dt={dt}")
plt.tight_layout()
plt.show()


   