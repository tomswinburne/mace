"""
Simple example usage of get_descriptors_gradients
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

# Get descriptors
descriptors = calc.get_descriptors(atoms)
print(f"Shape: {descriptors.shape}")  # (num_atoms, total_features)
print(np.abs(descriptors).mean())
# Get descriptor gradients (default: single axis with uniform weights)
gradients = calc.get_descriptors_gradients(atoms)
print(f"Shape: {gradients.shape}")  # (1, num_atoms, 3, total_features)
print(np.abs(gradients).mean())

# With custom weight tensor for multiple axes
weight_tensor = np.array([
    [2.0, 1.0, 1.0],  # Axis 0: weight oxygen more
    [1.0, 2.0, 1.0],  # Axis 1: weight first hydrogen more
])
gradients_weighted = calc.get_descriptors_gradients(atoms, weight_tensor=weight_tensor)
print(f"Weighted shape: {gradients_weighted.shape}")  # (2, num_atoms, 3, total_features)
print(np.abs(gradients_weighted).mean())
