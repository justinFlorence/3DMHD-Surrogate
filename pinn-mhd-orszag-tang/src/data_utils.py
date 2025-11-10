import h5py
import numpy as np
import os

def load_grid_and_state(grid_file, state_file):
    """
    Load x, y grid and 10-channel physical state data from AGATE Hall MHD Orszag–Tang benchmark files.
    """
    with h5py.File(grid_file, 'r') as fg:
        x = fg["subID0/x"][:, :, 0]
        y = fg["subID0/y"][:, :, 0]

    with h5py.File(state_file, 'r') as fs:
        data = fs["subID0/vector"][:, :, :, 0]  # shape (10, Ny, Nx)

    return x, y, data

def process_and_save(grid_file, state_file, output_file):
    """
    Load raw grid/state HDF5 files and save as compressed NumPy archive.
    """
    print(f"[INFO] Processing dataset from: {os.path.basename(grid_file)} and {os.path.basename(state_file)}")
    x, y, data = load_grid_and_state(grid_file, state_file)
    
    print(f"[INFO] Shapes — x: {x.shape}, y: {y.shape}, data: {data.shape}")
    
    np.savez_compressed(output_file, x=x, y=y, data=data)
    print(f"[INFO] Saved processed data to {output_file}")

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    grid_file = os.path.join(base_path, "raw/hallOT512.grid.h5")
    state_file = os.path.join(base_path, "raw/hallOT512.state_000000.h5")
    output_file = os.path.join(base_path, "processed/orszag_processed_data.npz")

    process_and_save(grid_file, state_file, output_file)

