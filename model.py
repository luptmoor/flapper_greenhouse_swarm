import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


state_names = ['u', 'w', 'theta', 'q', 'r', 'ld', 'f', 'Gamma', 'gamma']
dt = 0.01;

lw = 0.081;
zeta = 0.634;
wn = 40;
tau_f = 0.0796;
tau_r = 0.05;
bx = 0.00421;
lz = 0.0271;
bz = 0.000916;
I = 0.000126;
m = 0.0294;
g = 9.80665;
c1 = 0.0114;
c2 = -0.0449;
c_corr = 0.175;

Kp_theta = -0.711;
Kd_theta = -0.0654;
Kp_vx = 0.35
Kp_vz = 80

def dx(x, inp):
    u, w, theta, q, r, ld, f, G, gamma = x;
    vx_ref, vz_ref, r_ref = inp;

    u_abs, w_abs, _, _, _, _ = y(x)

    # Control law
    theta_ref = Kp_vx * (vx_ref - u_abs)
    f_ref = Kp_vz * (vz_ref - w_abs)
    gamma_ref = Kp_theta * (theta_ref - theta) + Kd_theta * (0 - q)
    d_r = (r_ref - r) / tau_r;

    # Actuator dynamics
    d_f = (f_ref - f) / tau_f; # works

    d_G = wn**2 * (gamma_ref - gamma) - 2 * zeta * wn * G; # works
    d_gamma = G; # works
    d_ld = lw * G * np.cos(gamma + c_corr * u); # works


    # Rigid Body dynamics, vertical flight works
    d_q = (     -bx * f * lz * (u - lz * q)          + bz * f * ld * (w - ld * q)                  - 2 * ld * (c1*f + c2)           ) / I
    d_theta = q

    d_u = - d_theta * w   - g * np.sin(theta)                     - bx * f / m * (u - lz * d_theta + d_ld);
    d_w =   d_theta * u   + g * np.cos(theta) - 2/m * (c1*f + c2) - bz * f / m * (w - ld * d_theta);

    return d_u, d_w, d_theta, d_q, d_r, d_ld, d_f, d_G, d_gamma;

# Input signal function
def inp(t):
    theta_ref = 10 / 57.3
    q_ref = 0
    f_ref = (m*g - 2*c2 )/ (2*c1) + 0

    vx_ref = 1.5;
    vz_ref = 0;
    r_ref = 0.2;
    return vx_ref, vz_ref, r_ref

# Time-dependent version for solve_ivp
def f_wrapped(t, x):
    return dx(x, inp(t))

def y(x):
    u, w, theta, q, r, ld, f, G, gamma = x;

    u_abs = - u * np.cos(theta) - w * np.sin(theta)
    w_abs = - w * np.cos(theta) + u * np.sin(theta)
    theta = theta / np.pi * 180
    q = q / np.pi * 180
    r = r / np.pi * 180

    return u_abs, w_abs, theta, q, r, ld





x0 = np.zeros(9)
x0[6] = (m*g - 2*c2 )/ (2*c1)

t_span = (0, 10+dt)
t_eval = np.linspace(*t_span, int(round((10 // dt)))+1)

sol = solve_ivp(f_wrapped, t_span, x0, t_eval=t_eval, method='RK45')
output = y(sol.y)


# x0 = np.zeros(8)
# x0[5] = (m*g - 2*c2 )/ (2*c1)

# w_list = []
# t_list = []
# t = 0
# while True:

#     sol = solve_ivp(f_wrapped, (t, t+dt), x0, t_eval=np.linspace(t, t+dt, 2), method='RK45')
#     x0 = sol.y[:, -1].flatten()
#     print(t)
#     print(x0)
#     output2 = y(x0)

    
#     w_list.append(output2[1])
#     t_list.append(t)

#     t += dt

#     if t >= 10:
#         break;


# plt.plot(t_list, w_list)
# plt.plot(t_eval, output[1])
# plt.show()





# Plot states
for i in range(6):
    plt.plot(sol.t, output[i], label=state_names[i])
#plt.plot(sol.t, lw * np.sin(sol.y[5]) + 0.01, label='ld calc')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.title("Simulation of 8-State Nonlinear System with 2 Inputs")
plt.grid()
plt.show()