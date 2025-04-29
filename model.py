import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


state_names = ['u', 'w', 'theta', 'q', 'r', 'ld', 'f', 'Gamma', 'gamma']
dt = 0.01;


class FlapperModel:
    def __init__(self, u0=0, w0=0, theta0=0, q0=0, r0=0, ld0=0, f0=0, G0=0, gamma0=0):
        
        # Physical parameters
        self.lw = 0.081;
        self.lz = 0.0271;
        self.bx = 0.00421;
        self.bz = 0.000916;
        self.I = 0.000126;
        self.m = 0.0294;
        self.g = 9.80665;

        # Flapping and thrust parameters
        self.tau_f = 0.0796;
        self.c1 = 0.0114;
        self.c2 = -0.0449;
        
        # Dihedral model
        self.zeta = 0.634;
        self.wn = 40;
        self.c_corr = 0.175;

        # Yaw rate model
        self.tau_r = 0.05;

        # Controller gains
        self.Kp_theta = -0.511;
        self.Kd_theta = -0.0654;
        self.Kp_vx = 0.35
        self.Kp_vz = 20

        # Initialise state
        self.x = np.array([u0, w0, theta0, q0, r0, ld0, f0, G0, gamma0])


    def dx(self, t, x):
        """
        State equation x_dot as a function of x and u
        """

        u, w, theta, q, r, ld, f, G, gamma = x;
        u_abs, w_abs, _, _, _, _, _, _, _ = self.to_output(x)

        # Control law
        theta_ref = self.Kp_vx * (self.vx_cmd - u_abs)
        f_ref = self.Kp_vz * (self.vz_cmd - w_abs) + (self.m * self.g - self.c2) / self.c1
        gamma_ref = self.Kp_theta * (theta_ref - theta) + self.Kd_theta * (0 - q)
        d_r = (self.r_cmd - r) / self.tau_r

        # Actuator dynamics
        d_f = (f_ref - f) / self.tau_f; # works

        d_G = self.wn**2 * (gamma_ref - gamma) - 2 * self.zeta * self.wn * G; # works
        d_gamma = G; # works
        d_ld = self.lw * G * np.cos(gamma + self.c_corr * u); # works


        # Rigid Body dynamics, vertical flight works
        d_q = (     -self.bx * f * self.lz * (u - self.lz * q)          + self.bz * f * ld * (w - ld * q)                  - 2 * ld * (self.c1*f + self.c2)           ) / self.I
        d_theta = q

        d_u = - d_theta * w   - self.g * np.sin(theta)                     - self.bx * f / self.m * (u - self.lz * d_theta + d_ld);
        d_w =   d_theta * u   + self.g * np.cos(theta) - 2/self.m * (self.c1*f + self.c2) - self.bz * f / self.m * (w - ld * d_theta);

        return d_u, d_w, d_theta, d_q, d_r, d_ld, d_f, d_G, d_gamma;


    def to_state(y):
        u_abs, w_abs, theta_deg, q_deg, r_deg, ld, f, G, gamma = y;

        r = r_deg / 180 * np.pi;
        q = q_deg / 180 * np.pi;
        theta = theta_deg / 180 * np.pi
        u = -np.cos(theta) * u_abs + np.sin(theta) * w_abs;
        w = -np.sin(theta) * u_abs - np.cos(theta) * w_abs;

        return u, w, theta, q, r, ld, f, G, gamma;


    def to_output(self, x):
        u, w, theta, q, r, ld, f, G, gamma = x;

        u_abs = - u * np.cos(theta) - w * np.sin(theta)
        w_abs = - w * np.cos(theta) + u * np.sin(theta)
        theta_deg = theta / np.pi * 180
        q_deg = q / np.pi * 180
        #r_deg = r / np.pi * 180

        return u_abs, w_abs, theta_deg, q_deg, r, ld, f, G, gamma

    def set_input(self, vx, vz, r):
        self.vx_cmd = vx;
        self.vz_cmd = vz;
        self.r_cmd = r;

    # Input signal function
    def inp(self, t):
        theta_ref = 10 / 57.3
        q_ref = 0
        f_ref = (self.m*self.g - 2*self.c2 )/ (2*self.c1) + 0

        vx_ref = 10;
        vz_ref = 10;
        r_ref = 0.2;
        return vx_ref, vz_ref, r_ref
    
    
    def advance(self, dt):
        sol = solve_ivp(self.dx, (0, dt), self.x, t_eval=np.linspace(0, dt, 2), method='RK45')
        self.x = sol.y[:, -1].flatten()


if __name__ == '__main__':
    model = FlapperModel();
    dt = 0.01;
    t = 0.0;
    output = np.zeros([1000, 10])

    for i in range(1, 1000):
        model.set_input(1, 1, 0.2);
        model.advance(dt)
        y = model.to_output(model.x)
        t += dt;

        output[i, :9] = y
        output[i, 9] = t
        

    for j in range(3):
        plt.plot(output[:, 9], output[:, j], label=state_names[j])
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.title("Simulation of 8-State Nonlinear System with 2 Inputs")
    plt.grid()
    plt.show()
    




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
