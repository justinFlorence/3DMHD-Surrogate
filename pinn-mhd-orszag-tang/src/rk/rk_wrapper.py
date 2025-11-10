# src/rk/rk_wrapper.py
import torch


def rk4_step(model, x, u0, dt):
    """
    Runge-Kutta 4th-order time stepping for evolving state u0.

    Parameters:
    - model: neural network model approximating du/dt = f(x, u)
    - x: spatial inputs (N, 2)
    - u0: initial state values at x (N, D)
    - dt: timestep (float)

    Returns:
    - u1: evolved state after timestep dt
    """
    x = x.requires_grad_(True)
    u0 = u0.requires_grad_(True)

    k1 = model(torch.cat([x, u0], dim=1))
    k2 = model(torch.cat([x, u0 + 0.5 * dt * k1], dim=1))
    k3 = model(torch.cat([x, u0 + 0.5 * dt * k2], dim=1))
    k4 = model(torch.cat([x, u0 + dt * k3], dim=1))

    u1 = u0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u1


def rk4_integrate(model, x, u0, dt, steps):
    """
    Integrate the system forward using RK4 for a number of time steps.

    Parameters:
    - model: neural net approximating du/dt = f(x, u)
    - x: spatial inputs (N, 2)
    - u0: initial state (N, D)
    - dt: timestep (float)
    - steps: number of integration steps

    Returns:
    - states: list of tensors [u0, u1, ..., uT]
    """
    states = [u0]
    u = u0
    for _ in range(steps):
        u = rk4_step(model, x, u, dt)
        states.append(u)
    return states


def euler_step(model, x, u0, dt):
    x = x.requires_grad_(True)
    u0 = u0.requires_grad_(True)
    return u0 + dt * model(torch.cat([x, u0], dim=1))

