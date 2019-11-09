import numpy as np
import random
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, n_k):
        self.n_k = n_k

    def pull_arm(self, i):
        return 0

    def max_mu(self):
        return 0


class Bernoulli(Environment):
    def __init__(self, array):
        super().__init__(n_k=len(array))
        self.mu = array

    def pull_arm(self, i):
        return np.random.binomial(1, self.mu[i])

    def max_mu(self):
        return np.max(self.mu)


class RandomBernoulli(Environment):
    def __init__(self, n_k):
        super().__init__(n_k)
        self.mu = np.random.uniform(0, 1, n_k)

    def pull_arm(self, i):
        return np.random.binomial(1, self.mu[i])

    def max_mu(self):
        return np.max(self.mu)


class Agent:
    def __init__(self, env):
        self.env = env
        self.total_reward = np.zeros(self.env.n_k)
        self.n_pulled = np.zeros(self.env.n_k)
        self.empirical_estimates = np.full(self.env.n_k, 1.)
        self.history = []
        self.steps = 0
        self.T = 0

    def run(self, steps):
        self.T = steps
        for i in range(steps):
            self.steps += 1
            k = self.choose_arm()
            reward = self.env.pull_arm(k)

            # Update held values
            self.total_reward[k] += reward
            self.n_pulled[k] += 1
            self.empirical_estimates[k] = self.total_reward[k] / self.n_pulled[k]

            self.history.append((k, reward))

    def choose_arm(self):
        return 0

    def compute_confidence_bounds(self):
        n_ts = np.maximum(self.n_pulled, np.full(self.env.n_k, 1))
        return np.sqrt(2 * np.log(self.T)) * np.full(self.env.n_k, 1.) / np.sqrt(n_ts)


class UniformSelection(Agent):
    def __init__(self, env):
        super().__init__(env)

    def choose_arm(self):
        return np.random.randint(0, self.env.n_k)


class EpsilonGreedy(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.epsilon = 1

    def choose_arm(self):
        if random.random() < self.epsilon:
            arm = np.random.randint(0, self.env.n_k)
        else:
            arm = np.argmax(self.empirical_estimates)

        self.epsilon = 1 / (self.steps ** (1/3))

        return arm


class UCB1(Agent):
    def __init__(self, env):
        super().__init__(env)

    def choose_arm(self):
        return np.argmax(self.empirical_estimates + self.compute_confidence_bounds())


class Thompson(Agent):
    def __init__(self, env):
        super().__init__(env)

    def choose_arm(self):

        ss = self.total_reward
        fs = self.n_pulled - self.total_reward

        probs = np.zeros(self.env.n_k)

        for i in range(self.env.n_k):
            probs[i] = np.random.beta(ss[i] + 1, fs[i] + 1)
        arm = np.argmax(probs)

        return arm


def plot_arm_choice(hs):
    xs = []
    ys = []
    s = 0
    for arm, _ in hs:
        s += 1
        ys.append(arm)
        xs.append(s)

    # Arm Choice
    plt.plot(xs,  ys, 'o')
    plt.axis([0, len(hs), 0, env.n_k])
    plt.show()


def plot_regret(hss, max_mu, labels):

    for hs in hss:
        xs = []
        ys = []
        s = 0
        total_regret = 0
        for _, reward in hs:
            s += 1
            total_regret += max_mu - reward
            ys.append(total_regret)
            xs.append(s)

        plt.plot(xs, ys)

    # plt.yscale('log')
    plt.xscale('log')
    plt.legend(labels)
    plt.xlabel('Time')
    plt.ylabel('Total regret')

    plt.show()


if __name__ == '__main__':
    # env = RandomBernoulli(10)
    env = Bernoulli([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])

    histories = []
    labels = []

    for agent, name in [(UCB1, 'UCB1'), (EpsilonGreedy, 'e-greedy'), (UniformSelection, 'Uniform'),
                        (Thompson, 'Thompson Sampling')]:
        a = agent(env)
        a.run(10000)
        histories.append(a.history)
        labels.append(name)

    # plot_arm_choice(agent.history)
    plot_regret(histories, env.max_mu(), labels=labels)
