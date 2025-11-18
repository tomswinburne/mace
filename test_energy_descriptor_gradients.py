"""
Test script for get_energy_descriptors_gradients function
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

# Test 1: Get energy-descriptor gradients
print("Test 1: Getting energy-descriptor gradients")
dE_dD = calc.get_energy_descriptors_gradients(atoms)
print(f"Shape: {dE_dD.shape}")
print(f"(num_atoms={len(atoms)}, total_features={dE_dD.shape[1]})")
print(f"Mean absolute value: {np.abs(dE_dD).mean():.6f}")
print()

# Test 2: Verify using chain rule: dE/dpos = dE/dD * dD/dpos
print("Test 2: Verifying using chain rule")
# Get forces (dE/dpos)
forces = atoms.get_forces()  # Shape: (num_atoms, 3)
print(f"Forces shape: {forces.shape}")

# Get descriptor gradients (dD/dpos)
dD_dpos = calc.get_descriptors_gradients(atoms)  # Shape: (num_atoms, 3, total_features)
print(f"Descriptor gradients shape: {dD_dpos.shape}")

# Chain rule: dE/dpos_ia = sum_j (dE/dD_ij * dD_ij/dpos_ia)
# where i is atom index, a is coordinate (x,y,z), j is descriptor feature
forces_from_chain = np.zeros((len(atoms), 3))
for atom_idx in range(len(atoms)):
    for coord_idx in range(3):
        # dE/dpos[atom_idx, coord_idx] = sum over features of dE/dD * dD/dpos
        forces_from_chain[atom_idx, coord_idx] = np.dot(
            dE_dD[atom_idx, :],  # dE/dD for this atom
            dD_dpos[atom_idx, coord_idx, :]  # dD/dpos for this atom and coordinate
        )

# Forces should be negative gradient of energy
forces_from_chain = -forces_from_chain

print(f"\nForces from ASE (first atom):")
print(forces[0])
print(f"\nForces from chain rule (first atom):")
print(forces_from_chain[0])
print(f"\nRelative difference: {np.linalg.norm(forces - forces_from_chain) / np.linalg.norm(forces):.6f}")
print()

# Test 3: Numerical verification - perturb a descriptor via positions
print("Test 3: Numerical gradient check")
eps = 1e-5
atom_idx = 0
coord_idx = 0

# Get baseline energy
E_0 = atoms.get_potential_energy()

# Perturb position slightly
atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[atom_idx, coord_idx] += eps
atoms_pert.positions = pos

# Get new energy
atoms_pert.calc = calc
E_1 = atoms_pert.get_potential_energy()

# Numerical gradient: dE/dpos
numerical_dE_dpos = (E_1 - E_0) / eps
print(f"Numerical dE/dpos[{atom_idx},{coord_idx}]: {numerical_dE_dpos:.6f}")

# Analytical gradient from chain rule
analytical_dE_dpos = np.dot(
    dE_dD[atom_idx, :],
    dD_dpos[atom_idx, coord_idx, :]
)
print(f"Analytical dE/dpos[{atom_idx},{coord_idx}]: {analytical_dE_dpos:.6f}")
print(f"Difference: {abs(numerical_dE_dpos - analytical_dE_dpos):.6e}")
print()

print("All tests completed!")
