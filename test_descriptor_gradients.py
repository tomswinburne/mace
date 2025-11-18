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

# Test 2: Get descriptor gradients with default weight_tensor (ones)
print("Test 2: Getting descriptor gradients (default weights)")
gradients = calc.get_descriptors_gradients(atoms)
print(f"Gradients shape: {gradients.shape}")
print(f"(num_axes={gradients.shape[0]}, num_atoms={len(atoms)}, 3, total_features={gradients.shape[3]})")
print()

# Test 3: Get descriptor gradients with custom weight_tensor
print("Test 3: Getting descriptor gradients (custom weight tensor)")
weight_tensor = np.array([
    [2.0, 1.0, 1.0],  # Axis 0: weight oxygen atom more
    [1.0, 2.0, 1.0],  # Axis 1: weight first H more
])
gradients_weighted = calc.get_descriptors_gradients(atoms, weight_tensor=weight_tensor)
print(f"Weighted gradients shape: {gradients_weighted.shape}")
print(f"(num_axes={gradients_weighted.shape[0]}, num_atoms={len(atoms)}, 3, total_features={gradients_weighted.shape[3]})")
print()

# Test 4: Check that gradients are different with different weights
print("Test 4: Comparing gradients with different weights")
print(f"Are gradients different with different weights? {not np.allclose(gradients[0], gradients_weighted[0])}")
print()

# Test 5: Get descriptor gradients for fewer layers
print("Test 5: Getting descriptor gradients for first layer only")
gradients_layer1 = calc.get_descriptors_gradients(atoms, num_layers=1)
print(f"Gradients shape (1 layer): {gradients_layer1.shape}")
print(f"(num_axes={gradients_layer1.shape[0]}, num_atoms={len(atoms)}, 3, total_features={gradients_layer1.shape[3]})")
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

# Analytical gradient (take first axis from default single-axis result)
analytical_grad_flat = gradients[0, atom_idx, coord_idx, :]

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
