import math
import matplotlib.pyplot as plt
import numpy as np


class NavierStokes:

    def __init__(self, dx, dy, rho, nu):
        """
        Args:
            dx: X Step Size, [m]
            dy: Y Step Size, [m]
            rho: Fluid Density, [kg/m^3]
            nu: Kineamtic Viscosity, [m^2/s]
        """

        # Fluid properties
        self.rho = rho
        self.nu = nu

        # Grid Bounds
        self.x_min = -0.01
        self.x_max = +0.01
        self.y_min = -0.01
        self.y_max = +0.01

        # Grid Spacing
        self.dx = dx
        self.dy = dy

        # Grid Size
        self.nx = math.floor((self.x_max - self.x_min) / dx) + 1
        self.ny = math.floor((self.y_max - self.y_min) / dy) + 1

        # Input Space
        self.x = np.linspace(self.x_min, self.x_min + (self.nx-1) * dx, self.nx)
        self.y = np.linspace(self.y_min, self.y_min + (self.ny-1) * dy, self.ny)

        # Output Space
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))

        # Residuals
        self.res_x = []
        self.res_y = []
        self.res_p = []


    def set_ICs(self):
        self.u[:, :] = 1.0
        self.v[:, :] = 0.0
        self.p[:, :] = 0.0


    def set_BCs(self):
        pass


    def solve(self):
        """
        """

        # u, v, p at iteration n+1
        u  = self.u
        v  = self.v
        p  = self.p

        # u, v, p at iteration n
        un = self.u.copy()
        vn = self.v.copy()
        pn = self.p.copy()

        # Simulation settings
        rho = self.rho
        nu = self.nu
        dx = self.dx
        dy = self.dy


        # Determine velocity field
        for it in range(10):
            for j in range(1, self.ny-1):
                for i in range(1, self.nx-1):

                    # X velocity component
                    u_num_p = (-1/rho) * (pn[i+1, j] - pn[i-1, j]) / (2*dx)

                    u_num_conv_x = un[i, j] * un[i-1, j] / dx
                    u_num_conv_y = vn[i, j] * un[i, j-1] / dy

                    u_num_diff_x = nu * (un[i+1, j] + un[i-1, j]) / (dx*dx)
                    u_num_diff_y = nu * (un[i, j+1] + un[i, j-1]) / (dy*dy)

                    u_den_conv_x = un[i, j] / dx
                    u_den_conv_y = vn[i, j] / dy

                    u_den_diff_x  = (2*nu) / (dx*dx)
                    u_den_diff_y  = (2*nu) / (dy*dy)

                    u[i, j] = (u_num_conv_x + u_num_conv_y + u_num_diff_x + u_num_diff_y + u_num_p) \
                            / (u_den_conv_x + u_den_conv_y + u_den_diff_x + u_den_diff_y)

                    # Y velocity component
                    v_num_p = (-1/rho) * (pn[i, j+1] - pn[i, j-1]) / (2*dy)

                    v_num_conv_x = un[i, j] * vn[i-1, j] / dx
                    v_num_conv_y = vn[i, j] * vn[i, j-1] / dy

                    v_num_diff_x = nu * (vn[i+1, j] + vn[i-1, j]) / (dx*dx)
                    v_num_diff_y = nu * (vn[i, j+1] + vn[i, j-1]) / (dy*dy)

                    v_den_conv_x = un[i, j] / dx
                    v_den_conv_y = vn[i, j] / dy

                    v_den_diff_x  = (2*nu) / (dx*dx)
                    v_den_diff_y  = (2*nu) / (dy*dy)

                    v[i, j] = (v_num_conv_x + v_num_conv_y + v_num_diff_x + v_num_diff_y + v_num_p) \
                            / (v_den_conv_x + v_den_conv_y + v_den_diff_x + v_den_diff_y)

            un = self.u.copy()
            vn = self.v.copy()

        # Determine residuals
        residual_x = np.zeros((self.nx, self.ny))
        residual_y = np.zeros((self.nx, self.ny))

        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):

                lhs_x = u[i, j] * (u[i, j] - u[i-1, j]) / dx \
                      + v[i, j] * (u[i, j] - u[i, j-1]) / dy \

                rhs_x = (-1/rho) * (p[i+1, j] - p[i-1, j]) / (2*dx) \
                      + nu * (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (dx*dx) \
                      + nu * (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (dy*dy)

                lhs_y = u[i, j] * (v[i, j] - v[i-1, j]) / dx \
                      + v[i, j] * (v[i, j] - v[i, j-1]) / dy \

                rhs_y = (-1/rho) * (p[i, j+1] - p[i, j-1]) / (2*dy) \
                      + nu * (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / (dx*dx) \
                      + nu * (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / (dy*dy)

                residual_x[i, j] = lhs_x - rhs_x
                residual_y[i, j] = lhs_y - rhs_y

        self.res_x.append(np.sqrt(np.mean(np.square(residual_x))))
        self.res_y.append(np.sqrt(np.mean(np.square(residual_y))))


        # Determine pressure field
        # Account for numerical divergence of velocity
        while True:

            # Setup next iteration
            pn = self.p.copy()

            for j in range(0, self.ny-1):
                for i in range(0, self.nx-1):

                    RHS = (-rho) * ((u[i+1, j] - u[i-1, j]) * (u[i+1, j] - u[i-1, j]) / (4*dx*dx) + \
                                    (u[i, j+1] - u[i, j-1]) * (v[i+1, j] - v[i-1, j]) / (4*dy*dx) + \
                                    (u[i, j+1] - u[i, j-1]) * (v[i+1, j] - v[i-1, j]) / (4*dy*dx) + \
                                    (v[i, j+1] - v[i, j-1]) * (v[i, j+1] - v[i, j-1]) / (4*dy*dy))

                    p[i, j] = (0.5/(dx*dx + dy*dy)) * ((p[i+1, j] + p[i-1, j]) * (dy*dy) + \
                                                    (p[i, j+1] + p[i, j-1]) * (dx*dx) + \
                                                    (-RHS) * (dx*dx * dy*dy))

            # Check if solution converged
            rms_diff = np.sqrt(np.mean(np.square(p - pn)))
            rms_p = np.sqrt(np.mean(np.square(p)))

            if rms_diff / rms_p < 1E-3:
                break


        # Determine Residuals
        residual_p = np.zeros((self.nx, self.ny))

        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):

                lhs_p = (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / (dx*dx) + \
                        (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / (dy*dy)

                rhs_p = (-rho) * ((u[i+1, j] - u[i-1, j]) * (u[i+1, j] - u[i-1, j]) / (4*dx*dx) + \
                                  (u[i, j+1] - u[i, j-1]) * (v[i+1, j] - v[i-1, j]) / (4*dy*dx) + \
                                  (u[i, j+1] - u[i, j-1]) * (v[i+1, j] - v[i-1, j]) / (4*dy*dx) + \
                                  (v[i, j+1] - v[i, j-1]) * (v[i, j+1] - v[i, j-1]) / (4*dy*dy))

                residual_p[i, j] = lhs_p - rhs_p

        self.res_p.append(np.sqrt(np.mean(np.square(residual_p))))


    def show_velocity(self):
        """
        """

        X, Y = np.meshgrid(self.x, self.y)
        plt.quiver(Y, X, self.u, self.v, scale = 50)

        # Setup axis label
        plt.xlabel('X, [m]')
        plt.ylabel('Y, [m]')

        # Show plot
        plt.show()


    def show_pressure(self):
        """
        """

        X, Y = np.meshgrid(self.x, self.y)
        plt.contour(Y, X, self.p, levels = 10) 
        plt.contourf(Y, X, self.p, alpha=0.5, levels = 10)
        plt.colorbar()

        # Setup axis label
        plt.xlabel('X, [m]')
        plt.ylabel('Y, [m]')

        # Show plot
        plt.show()


    def show_residuals(self):
        """
        """

        # Plot residuals
        plt.plot(range(len(self.res_x)), self.res_x, label="Residual U")
        plt.plot(range(len(self.res_y)), self.res_y, label="Residual V")
        plt.plot(range(len(self.res_p)), self.res_p, label="Residual P")

        # Setup axis label
        plt.xlabel("Iteration Number, [-]")
        plt.ylabel("Residual, [-]")

        # Setup axis scale
        plt.xscale('linear')
        plt.yscale('log')

        # Show plot
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":

    sim = NavierStokes(0.0005, 0.0005, 1.225, 1.5E-3)

    sim.u[:, -1] = 1.0
    sim.u[:, 0] = -1.0
    sim.v[-1, :] = -1.0
    sim.v[0, :] = 1.0

    # Show initial fields
    sim.show_velocity()
    sim.show_pressure()

    for it in range(32):
        print(f"Iteration {it}!")
        sim.solve()

    # Show solved fields
    sim.show_velocity()
    sim.show_pressure()

    # Show residuals
    sim.show_residuals()
