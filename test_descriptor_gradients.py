"""
Simple test script for get_descriptors_gradients function
"""
import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

# Create a simple water molecule
atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586]
    ]
)

# Initialize MACE calculator with the small MACE-MP-0 model
calc = mace_mp(model="small", device="cpu", default_dtype="float32")
atoms.calc = calc

# Test 1: Get descriptors
print("Test 1: Getting descriptors")
descriptors = calc.get_descriptors(atoms, invariants_only=True)
print(f"Descriptors shape: {descriptors.shape}")
print(f"(num_atoms={len(atoms)}, num_layers, num_invariant_features)")
print()

# Test 2: Get descriptor gradients with default weight_vector (ones)
print("Test 2: Getting descriptor gradients (default weights)")
gradients = calc.get_descriptors_gradients(atoms)
print(f"Gradients shape: {gradients.shape}")
print(f"(num_atoms={len(atoms)}, 3, total_features={gradients.shape[2]})")
print()

# Test 3: Get descriptor gradients with custom weight_vector
print("Test 3: Getting descriptor gradients (custom weights)")
weight_vector = np.array([2.0, 1.0, 1.0])  # Weight oxygen atom more
gradients_weighted = calc.get_descriptors_gradients(atoms, weight_vector=weight_vector)
print(f"Weighted gradients shape: {gradients_weighted.shape}")
print()

# Test 4: Check that gradients are different with different weights
print("Test 4: Comparing gradients with different weights")
print(f"Are gradients different with different weights? {not np.allclose(gradients, gradients_weighted)}")
print()

# Test 5: Get descriptor gradients for fewer layers
print("Test 5: Getting descriptor gradients for first layer only")
gradients_layer1 = calc.get_descriptors_gradients(atoms, num_layers=1)
print(f"Gradients shape (1 layer): {gradients_layer1.shape}")
print(f"(num_atoms={len(atoms)}, 3, total_features={gradients_layer1.shape[2]})")
print()

# Test 6: Numerical gradient check (sanity check)
print("Test 6: Numerical gradient check")
eps = 1e-5
atom_idx = 0  # Check oxygen atom
coord_idx = 0  # Check x-coordinate

# Get descriptors at current position - shape: (num_atoms, total_features)
desc_0_flat = calc.get_descriptors(atoms, invariants_only=True)
# Reshape to (num_atoms, num_layers, num_invariant_features)
desc_0 = desc_0_flat.reshape(len(atoms), 2, 128)  # 2 layers, 128 features
weighted_desc_0 = (np.ones(len(atoms))[:, None, None] * desc_0).sum(0)

# Perturb position
atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[atom_idx, coord_idx] += eps
atoms_pert.positions = pos

# Get descriptors at perturbed position
desc_1_flat = calc.get_descriptors(atoms_pert, invariants_only=True)
desc_1 = desc_1_flat.reshape(len(atoms), 2, 128)
weighted_desc_1 = (np.ones(len(atoms_pert))[:, None, None] * desc_1).sum(0)

# Numerical gradient - flatten to match output format
numerical_grad = (weighted_desc_1 - weighted_desc_0) / eps
numerical_grad_flat = numerical_grad.flatten()

# Analytical gradient
analytical_grad_flat = gradients[atom_idx, coord_idx, :]

print(f"Numerical gradient (first 5 features): {numerical_grad_flat[:5]}")
print(f"Analytical gradient (first 5 features): {analytical_grad_flat[:5]}")
abs_error = np.linalg.norm(numerical_grad_flat - analytical_grad_flat)
if np.linalg.norm(numerical_grad_flat) > 1e-10:
    rel_error = abs_error / np.linalg.norm(numerical_grad_flat)
    print(f"Relative error: {rel_error:.6e}")
else:
    print(f"Absolute error: {abs_error:.6e} (gradient near zero)")
print(f"Max absolute difference: {np.max(np.abs(numerical_grad_flat - analytical_grad_flat)):.6e}")
print()

print("All tests completed successfully!")
