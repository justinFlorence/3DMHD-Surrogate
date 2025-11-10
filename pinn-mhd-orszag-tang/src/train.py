# src/train.py
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.models.pinn import PINN
from src.losses.mhd_loss import mhd_loss
from src.rk.rk_wrapper import rk4_integrate
from src.data_utils import load_processed_data


def train():
    # Load data
    data_path = os.path.join("data", "processed", "orszag_data.pt")
    data = load_processed_data(data_path)
    x = data['x']          # (N, 2)
    u0 = data['u']         # (N, D) initial condition
    dx = data['dx']
    dy = data['dy']

    # Initialize model
    input_dim = x.shape[1] + u0.shape[1]  # x, y, and u as input to f(x, u)
    output_dim = u0.shape[1]              # returns du/dt
    model = PINN(input_dim, output_dim, hidden_layers=5, hidden_dim=128)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    # Training parameters
    epochs = 10000
    dt = 1e-3
    steps = 5  # RK steps per epoch (increases effective timespan)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        u_series = rk4_integrate(model, x, u0, dt, steps)
        uT = u_series[-1]  # Get the final evolved state

        loss = mhd_loss(x, uT, dx, dy)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.6e}")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/final_model.pt")


if __name__ == "__main__":
    train()

