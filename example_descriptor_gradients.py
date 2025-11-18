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
# Get descriptor gradients
gradients = calc.get_descriptors_gradients(atoms)
print(f"Shape: {gradients.shape}")  # (num_atoms, 3, total_features)
print(np.abs(gradients).mean())
# With custom weights
weight_vector = np.array([2.0, 1.0, 1.0])
gradients_weighted = calc.get_descriptors_gradients(atoms, weight_vector=weight_vector)
print(f"Weighted shape: {gradients_weighted.shape}")  # (num_atoms, 3, total_features)
