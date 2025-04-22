import numpy as np

lw = 0;
zeta = 0;
wn = 0;
tau = 0;
bx = 0;
lz = 0;
bz = 0;
I = 0;
m = 0;
g = 0;
c1 = 0;
c2 = 0;

def dx(x, inp):
    u, w, theta, f, G, g, ld = x;
    g_ref, f_ref = inp;

    d_ld = lw * G * np.cos(g);
    d_g = G;
    d_G = wn**2 * (g_ref - g) - 2 * zeta * wn * G;
    d_f = (f_ref - f) / tau;

    d_theta = (bx * lz * (f * u - d_ld) - bz * f * ld * w)     /    ( bx * lz**2 - bz * f * ld**2 - I);
    d_u = - d_theta * w - g * np.sin(theta) - bx * f / m * (u - lz * d_theta - d_ld);
    d_w = d_theta * u + g * np.cos(theta) = 2/m * (c1*f + c2) - bz * f / m * (w - ld * d_theta);