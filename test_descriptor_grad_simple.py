"""
Simple test to verify descriptor gradient computation
"""
import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

# Create a water molecule
atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]
)

# Initialize MACE calculator with float64 for better precision
calc = mace_mp(model="small", device="cpu", default_dtype="float64")

# Test 1: Single atom weighting
print("Test 1: Single weight vector")
weight_vector = np.array([1.0, 0.0, 0.0])  # Only atom 0
dD_dpos_single = calc.get_descriptors_gradients(atoms, weight_tensor=weight_vector.reshape(1, -1))
print(f"Shape: {dD_dpos_single.shape}")  # Should be (1, 3, 3, 256)

# Numerical verification
eps = 1e-6
atom_to_perturb = 0
coord_to_perturb = 2

desc_0 = calc.get_descriptors(atoms, invariants_only=True)
atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[atom_to_perturb, coord_to_perturb] += eps
atoms_pert.positions = pos
desc_1 = calc.get_descriptors(atoms_pert, invariants_only=True)

# Weighted sum with weight_vector
weighted_desc_0 = np.dot(weight_vector, desc_0)  # Sum over atoms
weighted_desc_1 = np.dot(weight_vector, desc_1)
numerical_grad = (weighted_desc_1 - weighted_desc_0) / eps

# Analytical
analytical_grad = dD_dpos_single[0, atom_to_perturb, coord_to_perturb, :]

print(f"\nNumerical gradient (first 10):\n{numerical_grad[:10]}")
print(f"Analytical gradient (first 10):\n{analytical_grad[:10]}")
print(f"Max difference: {np.max(np.abs(numerical_grad - analytical_grad)):.6e}")
print(f"Relative error: {np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad):.6e}")
