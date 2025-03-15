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
    
    def train(self, patterns, a=0.4, b=0.4):
        '''
        Training regime for low-activity patterns (which are more biologically plausible)
        Weight update equation taken from https://neuronaldynamics.epfl.ch/online/Ch17.S2.html#Ch17.E27

        INPUTS:
            patterns: Array containing multiple sqrt(N) x sqrt(N) pattern arrays
            a,b: Weight update equation constants (a = b is recommended for higher memory capacity)
        '''
        patterns = np.array(patterns)
        activity = a        # Target activity level
        b_const = b         # A constant between 0 and 1
        c_prime = 1 / (2 * activity * (1 - activity) * self.N)    # A constant > 0

        # Incorporate patterns into network weights
        for pattern in patterns:
            zeta = (pattern + 1) / 2    # Derived from p^{\mu}_i = 2 \zeta^{\mu}_i - 1
            zeta = np.array(zeta).reshape(-1, 1)
            self.W += np.dot(zeta - b_const, zeta.T - activity)
        
        # Zero out diagonal and multiply W by c_prime
        np.fill_diagonal(self.W, 0)
        self.W *= c_prime
    
    def forward(self, start_pattern, time_steps=500):
        '''
        INPUTS:
            start_pattern: sqrt(N) x sqrt(N) array
            time_steps: Simulation duration in milliseconds
        
        OUTPUT:
            firings_across_time: time_steps x 1 array of neurons that fired at each time step
        '''
        v = -65 * np.ones((self.N, 1))      # Initialize membrane potential
        u = self.b * -65                    # Initialize membrane recovery
        gamma = 25                               # Input current gaining factor

        # Convert starting pattern to numpy column vector
        start_pattern = np.array(start_pattern).reshape(-1, 1)

        firings_across_time = []
        fired: np.array

        for t in range(1, time_steps + 1):
            # External input current for all neurons at time t (using starting pattern)
            # Equation from https://www.seti.net/Neuron%20Lab/NeuronReferences/Izhikevich%20Model%20and%20backpropagation.pdf
            I = gamma * (start_pattern.T @ self.W).T

            # Update input currents using weights and membrane potentials of neurons that fired at time t-1
            if (t > 1) and (len(v[fired]) > 0):
                I += np.expand_dims(np.sum(self.W[:, fired] @ v[fired], axis = 1), axis = 1)

            # Update membrane potential `v` and recovery variable `u`
            # Note: for `v` we have to calculate in 0.5ms increments for numerical stability
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += self.a * (self.b * v - u)

            # When membrane potential `v` goes above 30 mV, we find the index and append it to `fired`
            # Then append `fired` to `firings_across_time`
            fired = np.where(v > 30)[0]
            if len(v[fired]) > 0:
                firings_across_time.append(fired)
            else:
                firings_across_time.append(np.empty(1))

            # Reset membrane potential/recovery of neurons that have fired
            for i in fired:
                v[i][0] = self.c[i][0]
                u[i][0] += self.d[i][0]

        return firings_across_time