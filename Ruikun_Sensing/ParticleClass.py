#This is the code of Gaussian Plume Generation.
import numpy as np

class Particle:
    def __init__(self, source_pos, params):

        """
        params: (1,4) param
        """

        self.q_ = params[0]           # emission rate (Q)
        self.xs_ = np.asarray(params[1:3], dtype=float)          # source location [x,y]
        self.layer_ = params[3]       # plume layer (usually 0)

    @staticmethod
    def to_ppm(c_g_per_m3, T=298.0, P=101325.0, M=16.04):
        """Convert methane concentration from g/m^3 to ppm (ideal gas, T in K, P in Pa, M in g/mol)."""
        Rg = 8.314  # J / (mol K)
        return c_g_per_m3 * Rg * T / (M * P) * 1e6

    #output of unit Guassian Plume model
    def getModelOutput(self, xm, wind, xs_override=None):

        """
        xm: (3,)    measuring location
        wind: (2,)  2D wind speed
        """
        single_mode = False
        if xs_override is None:
            xs = self.xs_.reshape(1, 2)
            single_mode = True
        else:
            xs = xs_override

        #normalized wind speed vector
        U = np.linalg.norm(wind) + 1e-8

        theta = np.arctan2(wind[1], wind[0])

        dx = xm[0] - xs[:, 0]
        dy = xm[1] - xs[:, 1]

        x_rel = dx * np.cos(theta) + dy * np.sin(theta)

        y_rel = -dx * np.sin(theta) + dy * np.cos(theta)


        # Derek Original Setting
        g = np.zeros_like(x_rel)
        mask = x_rel > 1e-3
        x_eff_valid = x_rel[mask]
        y_valid = y_rel[mask]

        sigma_y = np.maximum(0.1547 * np.power(x_eff_valid, 0.8932), 1e-6)
        sigma_z = np.maximum(0.1098 * np.power(x_eff_valid, 1.064), 1e-6)

        z = xm[2]
        H = 0
        Q = 1.0 #unit source emission rate

        g_valid = (
                Q
                / (2 * np.pi * U * sigma_y * sigma_z)
                * np.exp(-0.5 * (y_valid / sigma_y) ** 2)
                * (np.exp(-0.5 * ((z - H) / sigma_z) ** 2)
                   + np.exp(-0.5 * ((z + H) / sigma_z) ** 2))
        )

        #convert to ppm unit
        g[mask] = self.to_ppm(g_valid)

        if single_mode:
            return g[0]
        return g

    #output of the true emission source with emission rate Q
    def getTrueOutput(self, xm, u):
        return float(self.q_ * self.getModelOutput(xm, u))


