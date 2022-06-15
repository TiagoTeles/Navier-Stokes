import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class NonLinearConvection:

    def __init__(self, x_min, x_max, T, dx, dt):
        """
        Initialize to the LinearConvection1D class.

        Args:
            x_min: Left Boundary, [m]
            x_max: Right Boundary, [m]
            T: Simulation Duration, [s]
            dx: Spacial Step, [m]
            dt: Time Step, [m]
        """

        # Physical Variables
        self.x_min = x_min
        self.x_max = x_max
        self.T = T

        # Simulation Variables
        self.nx = math.floor((x_max - x_min) / dx) + 1
        self.nt = math.floor(T / dt) + 1
        self.dx = dx
        self.dt = dt

        # Data Structures
        self.t = np.linspace(  0.0, (self.nt-1) * dt, self.nt)
        self.x = np.linspace(x_min, (self.nx-1) * dx, self.nx)
        self.u = np.zeros((self.nx, self.nt))


    def set_ICs(self, function):
        """
        Set initial conditions for a pulse.
            function: Type of initial condition
        """

        if function == "Pulse":
            self.u[:, 0] = np.heaviside(self.x-1.0, 1.0) * np.heaviside(3.0-self.x, 1.0)

        if function == "Step":
            self.u[:, 0] = np.heaviside(self.x-2.0, 1.0)

        if function == "Exponential":
            self.u[:, 0] = np.exp(-(self.x-2.0)**2)

        if function == "Sine":
            self.u[:, 0] = np.sin(np.pi*self.x/(self.nx*self.dx))


    def set_BCs(self, value):
        """
        Set boundary conditions.

        Args:
            value: Amplitude of the Dirichlet BC, [-]
        """

        # Set u vector
        self.u[0, :] = value


    def solve(self):
        """
        Propagate solution by performing
        continuous matrix multiplication.
        """

        for n in range(self.nt-1):
            for i in range(1, self.nx):
                self.u[i, n+1] = self.u[i, n] * (1.0 - (self.dt/self.dx) * (self.u[i, n] - self.u[i-1, n]))


    def animate(self, n, line):
        """
        Set graph at timestep n.

        Args:
            n: Timestep
            line: Plot Graph

        Returns:
            line: Plot Graph
        """

        line.set_ydata(self.u[:, n])
        return line,


    def show(self, save=False):
        """
        Display solved system.

        Args:
            save: Save animation to .gif
        """

        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.u[:, 0])

        # Set labels
        ax.set_xlabel("x, [m]")
        ax.set_ylabel("u, [-]")

        # Show animation
        anim = animation.FuncAnimation(fig, self.animate, frames=self.nt, fargs=(line,),
                                       interval=1000*self.dt, blit=True)

        # Save animation
        if save is True:
            anim.save("Linear_Convection.gif")

        plt.show()


if __name__ == "__main__":

    sim = NonLinearConvection(0.0, 10.0, 10.0, 0.02, 0.02)

    # Set BCs and ICs
    sim.set_ICs("Exponential")
    sim.set_BCs(0.0)

    # Solve simulation
    sim.solve()

    # Plot simulation
    sim.show()
