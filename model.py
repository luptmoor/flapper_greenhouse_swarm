import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


state_names = ['u', 'w', 'theta', 'q', 'ld', 'f', 'Gamma', 'gamma']
dt = 0.01;

lw = 0.081;
zeta = 0.634;
wn = 40;
tau = 0.0796;
bx = 0.00421;
lz = 0.0271;
bz = 0.000916;
I = 0.000126;
m = 0.0294;
g = 9.80665;
c1 = 0.0114;
c2 = -0.0449;
c_corr = 0.175;

def dx(x, inp):
    u, w, theta, q, ld, f, G, gamma = x;
    gamma_ref, f_ref = inp;

    # Actuator dynamics
    d_f = (f_ref - f) / tau; # works

    d_G = wn**2 * (gamma_ref - gamma) - 2 * zeta * wn * G; # works
    d_gamma = G; # works
    d_ld = lw * G * np.cos(gamma + c_corr * u); # works


    # Rigid Body dynamics, vertical flight works
    d_q = (     -bx * f * lz * (u - lz * q)          + bz * f * ld * (w - ld * q)                  - 2 * ld * (c1*f + c2)           ) / I
    d_theta = q

    d_u = - d_theta * w   - g * np.sin(theta)                     - bx * f / m * (u - lz * d_theta + d_ld);
    d_w =   d_theta * u   + g * np.cos(theta) - 2/m * (c1*f + c2) - bz * f / m * (w - ld * d_theta);

    return d_u, d_w, d_theta, d_q, d_ld, d_f, d_G, d_gamma;

# Input signal function
def inp(t):
    gamma_ref = 0.01
    f_ref = (m*g - 2*c2 )/ (2*c1)
    return gamma_ref, f_ref

# Time-dependent version for solve_ivp
def f_wrapped(t, x):
    return dx(x, inp(t))

def y(x):
    u, w, theta, q, ld, f, G, gamma = x;

    u_abs = - u * np.cos(theta) - w * np.sin(theta)
    w_abs = - w * np.cos(theta) + u * np.sin(theta)

    return u_abs, w_abs, theta, q

x0 = np.zeros(8)
x0[5] = (m*g - 2*c2 )/ (2*c1)

t_span = (0, 1)
t_eval = np.linspace(*t_span, int(round((1 // dt))))

sol = solve_ivp(f_wrapped, t_span, x0, t_eval=t_eval, method='RK45')

output = y(sol.y)

# Plot states
for i in range(5):
    plt.plot(sol.t, sol.y[i], label=state_names[i])
#plt.plot(sol.t, lw * np.sin(sol.y[5]) + 0.01, label='ld calc')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.title("Simulation of 8-State Nonlinear System with 2 Inputs")
plt.grid()
plt.show()