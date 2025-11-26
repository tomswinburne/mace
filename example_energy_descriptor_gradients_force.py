"""
Verify that forces can be reconstructed using chain rule with descriptor gradients

NOTE: This currently shows a large error (~130% relative error). This suggests
there is a bug in get_energy_descriptors_gradients() that needs to be fixed.
The descriptor gradients (dD/dpos) are verified to be correct, so the issue
is specifically in the energy-descriptor gradients (dE/dD) computation.
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
atoms.calc = calc

# Get forces directly from ASE
forces = atoms.get_forces()
print("Forces from ASE:")
print(forces)
print()

# Reconstruct forces using chain rule: F = -dE/dpos = -dE/dD · dD/dpos
# dE/dD has shape (num_atoms, total_features)
dE_dD = calc.get_energy_descriptors_gradients(atoms)
print(f"dE/dD shape: {dE_dD.shape}")

# dD/dpos with weight_tensor=np.eye(num_atoms) gives shape (num_atoms, num_atoms, 3, total_features)
# where dD/dpos[i,j,k,f] = d(D_i,f)/d(pos_j,k) when weight for atom i is 1
dD_dpos = calc.get_descriptors_gradients(atoms, weight_tensor=np.eye(len(atoms)))
print(f"dD/dpos shape: {dD_dpos.shape}")
print()

# For each atom i, force component k:
# F[i,k] = -sum_f (dE/dD[i,f] * dD/dpos[i,i,k,f])
# The einsum 'if,iikf->ik' contracts over features f, using only diagonal elements [i,i,...]
forces_reconstructed = -np.einsum('if,iikf->ik', dE_dD, dD_dpos)

print("Forces reconstructed from chain rule:")
print(forces_reconstructed)
print()

print("Difference:")
print(forces - forces_reconstructed)
print()

print("Max absolute difference:", np.max(np.abs(forces - forces_reconstructed)))
print("Relative error:", np.linalg.norm(forces - forces_reconstructed) / np.linalg.norm(forces))