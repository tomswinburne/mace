"""
Debug force reconstruction to understand the error
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
atoms.calc = calc

# Get forces and energy
forces = atoms.get_forces()
energy = atoms.get_potential_energy()

print(f"Energy: {energy:.6f}")
print(f"Forces:\n{forces}\n")

# Numerical check: compute dE/dpos numerically
eps = 1e-5
numerical_forces = np.zeros((len(atoms), 3))

for atom_idx in range(len(atoms)):
    for coord_idx in range(3):
        atoms_pert = atoms.copy()
        atoms_pert.calc = calc
        pos = atoms_pert.positions.copy()
        pos[atom_idx, coord_idx] += eps
        atoms_pert.positions = pos

        energy_pert = atoms_pert.get_potential_energy()
        numerical_forces[atom_idx, coord_idx] = -(energy_pert - energy) / eps

print(f"Numerical forces:\n{numerical_forces}\n")
print(f"Force difference (analytical - numerical):\n{forces - numerical_forces}\n")

# Now test the chain rule
dE_dD = calc.get_energy_descriptors_gradients(atoms)
print(f"dE/dD shape: {dE_dD.shape}")
print(f"dE/dD mean: {np.abs(dE_dD).mean():.6f}")
print(f"dE/dD for atom 0 (first 5): {dE_dD[0, :5]}\n")

# Get descriptor gradients with identity weighting
dD_dpos = calc.get_descriptors_gradients(atoms, weight_tensor=np.eye(len(atoms)))
print(f"dD/dpos shape: {dD_dpos.shape}")

# Check: for a specific atom and coordinate, verify numerical gradient of descriptors
atom_idx = 0
coord_idx = 2  # z-coordinate
desc_0 = calc.get_descriptors(atoms, invariants_only=True)

atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[atom_idx, coord_idx] += eps
atoms_pert.positions = pos
desc_1 = calc.get_descriptors(atoms_pert, invariants_only=True)

numerical_dD_dpos = (desc_1 - desc_0) / eps  # Shape: (num_atoms, total_features)
analytical_dD_dpos = dD_dpos[atom_idx, atom_idx, coord_idx, :]  # For weight[atom_idx]=1

print(f"\nNumerical dD[atom={atom_idx}]/dpos[atom={atom_idx},coord={coord_idx}] (first 5):")
print(numerical_dD_dpos[atom_idx, :5])
print(f"Analytical (first 5):")
print(analytical_dD_dpos[:5])
print(f"Difference: {np.max(np.abs(numerical_dD_dpos[atom_idx] - analytical_dD_dpos)):.6e}\n")

# Now compute force using chain rule
forces_chain_rule = -np.einsum('if,iikf->ik', dE_dD, dD_dpos)
print(f"Forces from chain rule:\n{forces_chain_rule}\n")
print(f"Difference (ASE - chain rule):\n{forces - forces_chain_rule}\n")
print(f"Relative error: {np.linalg.norm(forces - forces_chain_rule) / np.linalg.norm(forces):.6f}")
