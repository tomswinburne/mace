"""
Simple example usage of get_energy_descriptors_gradients
"""
import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

# Create a water molecule
atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]
)

# Initialize MACE calculator
calc = mace_mp(model="small", device="cpu", default_dtype="float32")

# Get energy-descriptor gradients
dE_dD = calc.get_energy_descriptors_gradients(atoms)
print(f"Energy-descriptor gradients shape: {dE_dD.shape}")  # (num_atoms, total_features)
print(f"Mean absolute value: {np.abs(dE_dD).mean():.6f}")
print(f"Max absolute value: {np.abs(dE_dD).max():.6f}")
print(f"First atom, first 5 features: {dE_dD[0, :5]}")
