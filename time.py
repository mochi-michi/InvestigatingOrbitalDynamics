t = np.empty(n_steps + 1) 
out = np.empty((n_steps + 1, 4)) # columns for x, y, vx and vy
out[0] = y # initial condition
t[0] = 0.0 # initial time

for i in range(1, n_steps + 1): # time integration loop
    y = step_euler(y, dt)
    out[i] = y
    t[i] = i * dt

# Estimate period from y-crossings with vy>0
yvals = out[:, 1]
vyvals = out[:, 3]
idx = np.where((yvals[:-1] <= 0) & (yvals[1:] > 0) & (vyvals[1:] > 0))[0][0]
print(f"Estimated period ≈ {t[idx+1]:.4f} yr (Kepler {T:.4f} yr)")