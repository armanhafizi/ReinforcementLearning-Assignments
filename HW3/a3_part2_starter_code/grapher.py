import matplotlib.pyplot as plt
import numpy as np
import ast

def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":
    with open('DQN-data', 'r') as f:
            dqn_curves = ast.literal_eval(f.read())
    with open('C51-data', 'r') as f:
            c51_curves = ast.literal_eval(f.read())
    plot_arrays(dqn_curves, 'b', 'DQN')
    plot_arrays(c51_curves, 'r', 'C51')
    plt.legend(loc='best')
    plt.savefig('DQN-C51.png', dpi=320)
    plt.show()