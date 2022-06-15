import math
import matplotlib.pyplot as plt
import numpy as np


class Steady:

    def __init__(self, nx, ny, rho, nu):
        """
        Args:
            nx: X Grid Size, [-]
            ny: Y Grid Size, [-]
            rho: Fluid Density, [kg/m^3]
            nu: Kineamtic Viscosity, [m^2/s]
        """

        # Grid Bounds
        self.x_min = -1.0E3
        self.x_max = +1.0E3
        self.y_min = -1.0E3
        self.y_max = +1.0E3

        # Fluid properties
        self.rho = rho
        self.nu = nu

        # Grid Size
        self.nx = nx
        self.ny = ny

        # Grid Spacing
        self.dx = 2.0 / (nx-1)
        self.dy = 2.0 / (ny-1)

        # Input Space
        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)

        # Output Space
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))


    def set_ICs(self):
        """
        """
        pass


    def set_BCs(self):
        """
        """
        pass


    def solve(self):
        """
        """
        pass


    def show_velocity(self):
        """
        """

        X, Y = np.meshgrid(self.x, self.y)
        plt.quiver(Y, X, self.u, self.v, scale = 100)

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


if __name__ == "__main__":

    sim = Steady(21, 21, 1.225, 1.5E-3)

    # Show initial fields
    sim.show_velocity()
    sim.show_pressure()

    # Solve system
    sim.solve()

    # Show solved fields
    sim.show_velocity()
    sim.show_pressure()
