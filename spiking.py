import numpy as np
from numpy.typing import NDArray

''' ####   HOPFIELD NETWORK WITH SPIKING IZHIKEVICH NEURON MODELS  ####

Izhikevich model code adapted from https://medium.com/geekculture/the-izhikevich-neuron-model-fb5d953b41e5
and https://www.fabriziomusacchio.com/blog/2024-05-19-izhikevich_network_model

This network uses the parameters specified for modeling excitatory regular spiking neurons. '''

class SpikingHN:
    def __init__(self, N):
        self.N = N                          # Number of neurons
        self.S = np.zeros((N, N))           # Synaptic weight matrix

        self.r = np.random.rand(N, 1)       # Random factor
        self.a = 0.02 * np.ones((N, 1))     # Time scale of membrane recovery `u`
        self.b = 0.2 * np.ones((N, 1))      # Sensitivity of `u` to fluctuations in membrane potential `v`
        self.c = -65 + 15 * self.r**2       # After-spike reset value for `v`
        self.d = 8 - 6 * self.r**2          # After-spike reset value for `u`
    
    def train(self, patterns, a=0.4, b=0.4):
        '''
        Training regime for low-activity patterns (which are more biologically plausible)
        Weight update equation taken from https://neuronaldynamics.epfl.ch/online/Ch17.S2.html

        INPUTS:
            patterns: Array containing multiple sqrt(N) x sqrt(N) pattern arrays
            a,b: Weight update constants (a = b is recommended for symmetric weights and higher memory capacity)
        '''
        patterns = np.array(patterns)
        activity = a        # Target activity level
        b_const = b         # A constant between 0 and 1
        c_prime = 1 / (2 * activity * (1 - activity) * self.N)    # A constant > 0

        # Incorporate patterns into network weights
        for pattern in patterns:
            zeta = (pattern + 1) / 2    # Derived from p^{\mu}_i = 2 \zeta^{\mu}_i - 1
            zeta = np.array(zeta).reshape(-1, 1)
            self.S += np.dot(zeta - b_const, zeta.T - activity)
        
        # Zero out diagonal and multiply W by c_prime
        np.fill_diagonal(self.S, 0)
        self.S *= c_prime
    
    def forward_pattern(self, start_pattern, time_steps=250):
        '''
        INPUTS:
            start_pattern: sqrt(N) x sqrt(N) array
            time_steps: Simulation duration in milliseconds
        
        OUTPUT:
            firings_across_time: time_steps x 1 array of neurons that fired at each time step
        '''
        v = -65 * np.ones((self.N, 1))      # Initialize membrane potential
        u = self.b * -65                    # Initialize membrane recovery
        gamma = 25                               # Input current gain factor

        # Convert starting pattern to numpy row vector
        start_pattern = np.array(start_pattern).reshape(-1, 1).T

        firings_across_time: list[NDArray[np.intp]] = []
        firing_rates: NDArray[np.float64] = np.zeros(self.N)
        fired: NDArray[np.intp]

        for t in range(time_steps):
            # External input current for all neurons at time t (using starting pattern)
            # Equation from https://www.seti.net/Neuron%20Lab/NeuronReferences/Izhikevich%20Model%20and%20backpropagation.pdf
            I = gamma * (start_pattern @ self.S).T

            # Update input currents using weights and membrane potentials of neurons that fired at time t-1
            if (t > 0) and (len(v[fired]) > 0):
                I += np.expand_dims(np.sum(self.S[:, fired] @ v[fired], axis = 1), axis = 1)

            # Update membrane potential `v` and recovery variable `u`
            # Note: for `v` we have to calculate in 0.5ms increments for numerical stability
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += self.a * (self.b * v - u)

            # When membrane potential `v` goes above 30 mV, we find the index and append it to `fired`
            # Then append `fired` to `firings_across_time`
            fired = np.where(v > 30)[0]
            firings_across_time.append(fired)

            # Reset membrane potential/recovery of neurons that have fired
            for i in fired:
                firing_rates[i] += 1
                v[i][0] = self.c[i][0]
                u[i][0] += self.d[i][0]

        # Calculate firing rates
        firing_rates /= (time_steps / 1000)
        firing_rates = firing_rates.astype(int)

        return firings_across_time, firing_rates

    def forward(self, patterns):
        grayscale_outputs = []
        binarized_outputs = []

        for pattern in patterns:
            _, output = self.forward_pattern(pattern)
            grayscale_outputs.append(output)

            # Binarize pattern by setting black/white threshold at 10% of max firing rate in data
            binarized = np.where(output > (0.10 * np.max(output)), 255, 0)
            binarized_outputs.append(binarized)

        return np.array(grayscale_outputs), np.array(binarized_outputs)
