import torch
import torch.nn as nn


def divergence(field, coords):
    """Compute divergence of a vector field."""
    grad_outputs = torch.ones_like(field[:, 0])
    div = 0.0
    for i in range(coords.shape[1]):
        grad = torch.autograd.grad(
            field[:, i], coords, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0][:, i]
        div += grad
    return div


def gradient(scalar, coords):
    """Compute gradient of scalar field."""
    grad = torch.autograd.grad(
        scalar, coords, grad_outputs=torch.ones_like(scalar),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad


def compute_mhd_residuals(preds, coords, gamma=5/3):
    """
    Compute the residuals of the ideal MHD equations:
    - Continuity
    - Momentum (x and y)
    - Induction (B field)
    - Energy
    """
    rho = preds[:, 0:1]           # Density
    vx = preds[:, 1:2]           # Velocity x
    vy = preds[:, 2:3]           # Velocity y
    Bx = preds[:, 3:4]           # Magnetic field x
    By = preds[:, 4:5]           # Magnetic field y
    P  = preds[:, 5:6]           # Pressure

    v = torch.cat([vx, vy], dim=1)
    B = torch.cat([Bx, By], dim=1)
    
    # Total Energy E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
    E = P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2) + 0.5 * (Bx**2 + By**2)

    # Compute time derivatives
    dt_rho = gradient(rho, coords)[:, 2:3]
    dt_rhovx = gradient(rho * vx, coords)[:, 2:3]
    dt_rhovy = gradient(rho * vy, coords)[:, 2:3]
    dt_Bx = gradient(Bx, coords)[:, 2:3]
    dt_By = gradient(By, coords)[:, 2:3]
    dt_E = gradient(E, coords)[:, 2:3]

    # Spatial derivatives
    dvx_dx = gradient(vx, coords)[:, 0:1]
    dvy_dy = gradient(vy, coords)[:, 1:2]
    dBx_dx = gradient(Bx, coords)[:, 0:1]
    dBy_dy = gradient(By, coords)[:, 1:2]
    
    div_v = dvx_dx + dvy_dy
    div_B = dBx_dx + dBy_dy

    # Residuals
    continuity = dt_rho + rho * div_v

    momentum_x = dt_rhovx + gradient(P + 0.5 * (Bx**2 + By**2), coords)[:, 0:1] \
                 - (Bx * gradient(Bx, coords)[:, 0:1] + By * gradient(Bx, coords)[:, 1:2])

    momentum_y = dt_rhovy + gradient(P + 0.5 * (Bx**2 + By**2), coords)[:, 1:2] \
                 - (Bx * gradient(By, coords)[:, 0:1] + By * gradient(By, coords)[:, 1:2])

    induction_x = dt_Bx + gradient(vx * By - vy * Bx, coords)[:, 1:2]
    induction_y = dt_By - gradient(vx * By - vy * Bx, coords)[:, 0:1]

    energy = dt_E + divergence((E + P + 0.5 * (Bx**2 + By**2)) * v - (Bx * vx + By * vy) * B, coords)

    return continuity, momentum_x, momentum_y, induction_x, induction_y, energy, div_B


class MHDLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weights = weights or [1.0]*7

    def forward(self, preds, coords):
        continuity, momx, momy, ind_x, ind_y, energy, div_B = compute_mhd_residuals(preds, coords)

        loss = self.weights[0]*self.mse(continuity, torch.zeros_like(continuity)) + \
               self.weights[1]*self.mse(momx, torch.zeros_like(momx)) + \
               self.weights[2]*self.mse(momy, torch.zeros_like(momy)) + \
               self.weights[3]*self.mse(ind_x, torch.zeros_like(ind_x)) + \
               self.weights[4]*self.mse(ind_y, torch.zeros_like(ind_y)) + \
               self.weights[5]*self.mse(energy, torch.zeros_like(energy)) + \
               self.weights[6]*self.mse(div_B, torch.zeros_like(div_B))

        return loss

