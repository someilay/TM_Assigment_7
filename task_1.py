import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin, pi, exp
from typing import Callable


# Constants
M_1 = 1
M_2 = 3
M_3 = 2
B = 1e-3
alpha = beta = pi / 6
g = 10


# Intermediate constants
A_1 = g * ((M_1 + M_2 + 2 * M_3) * sin(alpha) - M_1 * sin(alpha + beta) * cos(beta)) / \
    (M_1 * (sin(beta)) ** 2 + M_2 + 4 * M_3)
A_2 = (-B) / (M_1 * (sin(beta)) ** 2 + M_2 + 4 * M_3)
A_3 = g * ((M_1 + M_2 + 4 * M_3) * sin(alpha + beta) - (M_1 + M_2 + 2 * M_3) * sin(alpha) * cos(beta)) / \
    (M_1 * (sin(beta)) ** 2 + M_2 + 4 * M_3)
A_4 = (B * cos(beta)) / (M_1 * (sin(beta)) ** 2 + M_2 + 4 * M_3)


# Time range
T_0 = 0
T_F = 2
STEPS = 200


def x(t: float) -> float:
    return (A_1 / (A_2 ** 2)) * exp(A_2 * t) - (A_1 / A_2) * t - A_1 / (A_2 ** 2)


def xi(t: float) -> float:
    return ((A_1 * A_4) / (A_2 ** 3)) * exp(A_2 * t) \
        + 0.5 * (A_3 - (A_1 * A_4) / A_2) * t ** 2 \
        + (3 - (A_1 * A_4) / (A_2 ** 2)) * t \
        - (A_1 * A_4) / (A_2 ** 3)


def v_x(t: float) -> float:
    return (A_1 / A_2) * exp(A_2 * t) - A_1 / A_2


def v_xi(t: float) -> float:
    return ((A_1 * A_4) / (A_2 ** 2)) * exp(A_2 * t) \
        + (A_3 - (A_1 * A_4) / A_2) * t \
        + (3 - (A_1 * A_4) / (A_2 ** 2))


def a_x(t: float) -> float:
    return A_1 * exp(A_2 * t)


def a_xi(t: float) -> float:
    return ((A_1 * A_4) / A_2) * exp(A_2 * t) + (A_3 - (A_1 * A_4) / A_2)


# Plot functions
def plot(funcs: list[Callable[[float], float]], colors: list[str], labels: list[str], save_to: str):
    ts = np.linspace(T_0, T_F, STEPS)
    res = [np.array([func(t) for t in ts]) for func in funcs]

    plt.title(''.join(labels).replace('$$', ','))
    for sub_res, color, label in zip(res, colors, labels):
        plt.plot(ts, sub_res, color, label=label, linewidth=1.0)
    plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.grid(True)
    plt.xlabel(r'$t$')
    plt.legend(loc='best')
    plt.savefig(save_to)
    plt.show()


def main():
    plot([x, xi], ['b', 'r'], ['$x(t)$', '$\\xi(t)$'], 'x_and_xi_from_t.png')
    plot([v_x, v_xi], ['b', 'r'], ['$\\dot{x}(t)$', '$\\dot{\\xi}(t)$'], 'vx_and_vxi_from_t.png')
    plot([a_x, a_xi], ['b', 'r'], ['$\\ddot{x}(t)$', '$\\ddot{\\xi}(t)$'], 'ax_and_axi_from_t.png')


if __name__ == '__main__':
    main()
