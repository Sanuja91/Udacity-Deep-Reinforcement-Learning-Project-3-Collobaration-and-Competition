import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, params):
        """Initialize parameters and noise process."""

        mu = params['mu']
        theta = params['theta']
        sigma = params['sigma']
        seed = params['seed']
        size = params['action_size']
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed.next())
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def create_noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class GaussianExploration:
    """Adds Gaussian Noise to the actions with the epsilon rate being annealed considering
       a x episode patience factor to execute decay"""
    def __init__(self, params):
        self.epsilon = params['max_epsilon']
        self.min_epsilon = params['min_epsilon']
        self.patience_episodes = params['patience_episodes']
        self.episodes_since_best_score = 0
        self.best_score = 0

        if params['decay_epsilon']:
            self.decay_rate = params['decay_rate']
        else:
            self.decay_rate = 1

    def create_noise(self, shape):
        return np.random.normal(0, 1, shape) * self.epsilon

    def decay(self, score):
        if score > self.best_score:
            self.best_score = score
            self.episodes_since_best_score = 0
        else:
            self.episodes_since_best_score += 1
            self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)     