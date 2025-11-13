import numpy as np
import matplotlib.pyplot as plt

# constants given in question statment + researched constants 
G = 4 * np.pi**2          
M_sun = 1.0               
a = 0.387   # semi major axis (AU)
e = 0.2056  # eccentricity (0 = circle, <1= ellipse)
r_peri = a * (1 - e)    # perihelion distance (AU)



# initial postion and velocity of mercury (ccw orbit)
x0 = r_peri
y0 = 0.0
vx0 = 0.0
vy0 = 12.0                # AU/year (given) (-12 for cw orbit)


# simulation time
dt = 0.001    # duration of each time step to advance simulation (measured in yrs)
t_max = 5.0     # simulate for this many years 
N_steps = int(t_max / dt)  
print(f'Will simulate {N_steps} time steps') 


# store the postion and velocity at each time step
x = np.zeros(N_steps)
y = np.zeros(N_steps)
vx = np.zeros(N_steps)
vy = np.zeros(N_steps)

# assign intital position and velocity 
x[0] = x0
y[0] = y0
vx[0] = vx0
vy[0] = vy0

# euler method loop
for n in range(N_steps - 1): # since n=1 is already computed inside

    # magnitude of position vec 
    r = np.sqrt(x[n]**2 + y[n]**2) # distance from sun to mercury
    
    # eqns 4-5 in script: gravitational acceleration
    ax = -G * M_sun * x[n] / r**3
    ay = -G * M_sun * y[n] / r**3

    # eqns 6-7: update position
    x[n+1] = x[n] + vx[n] * dt
    y[n+1] = y[n] + vy[n] * dt
    
    # eqns 8-9: update velocity
    vx[n+1] = vx[n] + ax * dt
    vy[n+1] = vy[n] + ay * dt

#plot
plt.figure(figsize=(6, 6))
plt.plot(x, y, label="Mercury orbit (Euler)")
plt.plot(0, 0, 'yo', markersize=10, label="Sun")
plt.scatter([x0], [y0], color='red', label="Start (perihelion)")
plt.axis('equal')
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
#plt.legend()
plt.title("Mercury Orbit (Euler Method)")
plt.grid(True)
plt.show()

'''
- the orbit visulation, for the el method, wont close and will continously spiral 
outward (if E_T increases) and inwards (if E_T decreases)
- smaller dt does improve this
- notice for higher dt, the orbit is incomplete (unstable)
'''