import numpy as np
import matplotlib.pyplot as plt


# Euler method computed in a function
def euler_method(x0, y0, vx0, vy0, dt, t_max, G=4*np.pi**2, M=1.0):

    N_steps = int(t_max / dt) # re assign locally
    
    # store the position and velocity at each time step
    x = np.zeros(N_steps)
    y = np.zeros(N_steps)
    vx = np.zeros(N_steps)
    vy = np.zeros(N_steps)

    # assign initial position and velocity 
    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0

    # euler method loop
    for n in range(N_steps - 1): # since n=0 is already computed

        # magnitude of position vec 
        r = np.sqrt(x[n]**2 + y[n]**2) # distance from sun to mercury
        
        # eqns 4-5 in script: gravitational acceleration
        ax = -G * M * x[n] / r**3
        ay = -G * M * y[n] / r**3

        # eqns 6-7: update position
        x[n+1] = x[n] + vx[n] * dt
        y[n+1] = y[n] + vy[n] * dt
        
        # eqns 8-9: update velocity
        vx[n+1] = vx[n] + ax * dt
        vy[n+1] = vy[n] + ay * dt

    t = np.linspace(0, t_max, N_steps)
    return t, x, y, vx, vy # returned arrays in (AU)


# constants given in question statement + researched constants 
G = 4 * np.pi**2          
M_sun = 1.0               
a = 0.387   # semi major axis (AU)
e = 0.2056  # eccentricity (0 = circle, <1= ellipse)
r_peri = a * (1 - e)    # perihelion distance (AU)



# initial position and velocity of mercury (ccw orbit)
x0 = r_peri
y0 = 0.0
vx0 = 0.0
vy0 = 12.0                # AU/year (given) (-12 for cw orbit)


# simulation time
dt = 0.0001    # duration of each time step to advance simulation (measured in yrs)
t_max = 5.0     # simulate for this many years 
N_steps = int(t_max / dt)  
print(f'Will simulate {N_steps} time steps') 


# run function to return relevant arrays
t, x, y, vx, vy = euler_method(x0, y0, vx0, vy0, dt, t_max, G=G, M=M_sun)
print(t, x, y, vx, vy)


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
- the orbit visualization, for the Euler method, won't close and will continuously spiral 
  outward (if E_T increases) and inwards (if E_T decreases)
- smaller dt does improve this
- notice for higher dt, the orbit is incomplete (unstable)
- for creativity marks, could 
'''


# comparing theoretical and measured time period
# find distance between sun and mercury at every time step
r_list = [] # values should go up (perihelion) and down (aphelion)
for i in range(len(x)):
    r = np.sqrt(x[i]**2 +y[i]**2)
    r_list.append(r)
    
# look for perihelion (closest points) where r goes down then up
# find record the corresponding time at each perhilion point
perihelion_times = []
for i in range(1, len(r_list) - 1):
    if r_list[i] < r_list[i-1] and r_list[i] < r_list[i+1]: # accept if value below < current value < value above
        perihelion_times.append(t[i])

print("\nOrbital Period Check")
print(f'There are {len(perihelion_times)} perihelion passages')


# find orbital period from time array
# find time between two consecutive times (at the perihelion points), this will be an interval
periods = []
for i in range(1, len(perihelion_times)):
    period = perihelion_times[i] - perihelion_times[i-1]
    periods.append(period)
        
    # store these intervals in an array and take average to find the average orbital period
    avg_period = sum(periods) / len(periods)
    theoretical = np.sqrt(a**3)
    
print(f"The average simulated period is {round(avg_period, 4)} years")
print(f"The calculated theoretical period is sqrt(a^3): { round(theoretical, 4)} years")
print(f"The difference is {round(avg_period - theoretical, 5)} years")
    