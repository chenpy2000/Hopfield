import numpy as np

#### HOPFIELD NETWORK WITH SPIKING IZHIKEVICH NEURON MODELS ####

## Izhikevich model code adapted from https://medium.com/geekculture/the-izhikevich-neuron-model-fb5d953b41e5
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
        self.W /= c_prime
    
    def forward(self, start_pattern, time_steps=250):
        '''
        INPUTS
            start_pattern: sqrt(N) x sqrt(N) array with values of 1 and -1
            time_steps: number of milliseconds
        '''
        v = -65 * np.ones((self.N, 1))      # Initialize membrane potential
        u = self.b * -65                    # Initialize membrane recovery
        gamma = 1                              # Input current gaining factor

        # Convert -1's in starting pattern to 0's because we're not sure if neurons like negative inputs
        # Then convert it to numpy column vector
        start_pattern = np.where(start_pattern == -1, 0, start_pattern)
        start_pattern = np.array(start_pattern).reshape(-1, 1)

        firings_across_time = []
        voltage_across_time = []

        for t in range(1, time_steps + 1):
            # Configure input currents for all neurons
            if t == 1:
                # Initial external input (the starting pattern)
                # Equation from https://www.seti.net/Neuron%20Lab/NeuronReferences/Izhikevich%20Model%20and%20backpropagation.pdf
                I = gamma * (start_pattern.T @ self.W)
                # print(f"I shape: {np.shape(I)}")
            else:
                # External input at all other time steps (just noise)
                I = 0.1 * np.random.rand(1, self.N)

            # When membrane potential `v` goes above 30 mV, we find the index, and append it to `fired`,
            # then reset `v` and membrane recovery variable `u`
            fired = np.where(v > 30)
            firings_across_time.append(fired[0])
            voltage_across_time.append(float(v[10]))

            # Reset membrane potential/recovery of neurons that have fired
            for i in fired[0]:
                v[i] = self.c[i]
                u[i] += self.d[i]
            
            # Update input currents using weights and membrane potentials of fired neurons
            # Inspired by https://www.fabriziomusacchio.com/blog/2024-05-19-izhikevich_network_model/#input-current
            # print(f"v_fired shape: {np.shape(v[fired[0]])}")
            # print(f"W fired shape: {np.shape(self.W[:, fired[0]])}")
            if len(v[fired[0]]) > 0:
                I += np.expand_dims(np.sum(v[fired[0]].T @ self.W[:, fired[0]], axis = 1), axis = 1)
            else:
                I += np.expand_dims(np.sum(self.W[:, fired[0]], axis = 1), axis = 1).T

            # Update membrane potential `v` and recovery var `u`
            # Note: for `v` we have to do 0.5ms increments for numerical stability
            # print(f"v shape: {np.shape(v)}")
            # print(f"u shape: {np.shape(u)}")
            # print(f"I.T shape: {np.shape(I.T)}")
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I.T)
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I.T)
            u = u + self.a * (self.b * v - u)

        # output_pattern = np.where(v > 30, 1, -1)[0]      
        voltage_across_time = np.array(voltage_across_time)
        return firings_across_time, voltage_across_time