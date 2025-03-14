import numpy as np

#### HOPFIELD NETWORK WITH SPIKING IZHIKEVICH NEURON MODELS ####

## Izhikevich model code adapted from https://medium.com/geekculture/the-izhikevich-neuron-model-fb5d953b41e5
## and https://www.fabriziomusacchio.com/blog/2024-05-19-izhikevich_network_model
## (but with inhibitory neuron stuff removed)

class SpikingHN:
    def __init__(self, N):
        self.N = N                          # Number of neurons
        self.W = np.zeros((N, N))           # Weight matrix (variable "S" in original Izhikevich formulation)

        self.r = np.random.rand(N, 1)       # Random factor
        self.a = 0.02 * np.ones((N, 1))     # Time scale of membrane recovery `u`
        self.b = 0.2 * np.ones((N, 1))      # Sensitivity of `u` to fluctuations in membrane potential `v`
        self.c = -65 + 15 * self.r**2       # After-spike reset value for `v`
        self.d = 8 - 6 * self.r**2          # After-spike reset value for `u`
    
    def train(self, patterns, a=0.35, b=0.35):
        ## Training regime for low-activity patterns (which are more biologically plausible)
        ## Weight update equation taken from https://neuronaldynamics.epfl.ch/online/Ch17.S2.html#Ch17.E27
        patterns = np.array(patterns)
        activity = a        # Target activity level
        b_const = b         # A constant between 0 and 1
        c_prime = 1 / (2 * activity * (1 - activity) * self.N)    # A constant > 0

        # Incorporate patterns into network weights
        for pattern in patterns:
            zeta = (pattern + 1) / 2    # Derived from p^{\mu}_i = 2 \zeta^{\mu}_i - 1
            zeta = np.array(zeta).reshape(-1, 1)
            self.W += np.dot(zeta - b_const, zeta.T - activity)
        
        # Zero out diagonal and divide W by c_prime
        np.fill_diagonal(self.W, 0)
        self.W *= c_prime
    
    def forward(self, start_pattern, time_steps=50):
        '''
        INPUTS
            start_pattern: sqrt(N) x sqrt(N) array with values of 1 and -1
            time_steps: number of milliseconds
        '''
        v = -65 * np.ones((self.N, 1))      # Initialize membrane potential
        u = self.b * -65                    # Initialize membrane recovery
        gamma = 25                               # Input current gaining factor

        # Convert starting pattern to numpy column vector
        start_pattern = np.array(start_pattern).reshape(-1, 1)

        firings_across_time = []
        voltage_across_time = []
        fired: np.array

        for t in range(1, time_steps + 1):
            # Configure input currents for all neurons at time t
            if t == 1:
                # Initial external input (the starting pattern)
                # Equation from https://www.seti.net/Neuron%20Lab/NeuronReferences/Izhikevich%20Model%20and%20backpropagation.pdf
                I = gamma * (start_pattern.T @ self.W).T
            else:
                # External input at all other time steps (just noise)
                I = np.random.rand(self.N, 1)

            # Update input currents using weights and membrane potentials of neurons that fired at time t-1
            if t > 1:
                if len(v[fired]) > 0:
                    print(f"time {t}: \t{fired}")
                    I += np.expand_dims(np.sum(self.W[:, fired] @ v[fired], axis = 1), axis = 1)

            # Update membrane potential `v` and recovery var `u`
            # Note: for `v` we have to do 0.5ms increments for numerical stability
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += self.a * (self.b * v - u)

            # When membrane potential `v` goes above 30 mV, we find the index, and append it to `fired`,
            # then reset `v` and membrane recovery variable `u`
            fired = np.where(v > 30)[0]
            firings_across_time.append(fired)
            voltage_across_time.append(float(v[10]))

            # Reset membrane potential/recovery of neurons that have fired
            for i in fired:
                v[i][0] = self.c[i][0]
                u[i][0] += self.d[i][0]

        # output_pattern = np.where(v > 30, 1, -1)[0]      
        voltage_across_time = np.array(voltage_across_time)
        return firings_across_time, voltage_across_time