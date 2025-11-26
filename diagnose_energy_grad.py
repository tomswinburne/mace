"""
Detailed diagnosis of get_energy_descriptors_gradients
"""
import numpy as np
import torch
from ase import Atoms
from mace.calculators import mace_mp

atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]
)

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
atoms.calc = calc

print("=" * 80)
print("PART 1: Understand the model architecture")
print("=" * 80)

model = calc.models[0]
batch = calc._atoms_to_batch(atoms)

# Need training=True to get contributions, and we need gradients
out = model(batch.to_dict(), compute_stress=False, training=True)

print(f"Total energy: {out['energy'].item():.6f}")
print(f"Output keys: {out.keys()}")
if 'contributions' in out:
    print(f"Energy contributions shape: {out['contributions'].shape}")
    print(f"Energy contributions:\n{out['contributions']}")
    print(f"Sum of contributions: {out['contributions'].sum().item():.6f}")

    # Check each contribution
    print(f"\nContribution 0 (E0 + pair?): {out['contributions'][0, 0].item():.6f}")
    print(f"Contribution 1 (layer 0 readout?): {out['contributions'][0, 1].item():.6f}")
    print(f"Contribution 2 (layer 1 readout?): {out['contributions'][0, 2].item():.6f}")
else:
    print("Model doesn't return 'contributions' - will focus on total energy")

print("\n" + "=" * 80)
print("PART 2: Check if E0 and pair energies depend on positions")
print("=" * 80)

# Perturb position slightly
eps = 1e-6
atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[0, 2] += eps
atoms_pert.positions = pos

batch_pert = calc._atoms_to_batch(atoms_pert)
out_pert = model(batch_pert.to_dict(), compute_stress=False, training=True)

if 'contributions' in out:
    print(f"Delta contribution 0: {(out_pert['contributions'][0, 0] - out['contributions'][0, 0]).item():.6e}")
    print(f"Delta contribution 1: {(out_pert['contributions'][0, 1] - out['contributions'][0, 1]).item():.6e}")
    print(f"Delta contribution 2: {(out_pert['contributions'][0, 2] - out['contributions'][0, 2]).item():.6e}")
print(f"Delta total energy: {(out_pert['energy'] - out['energy']).item():.6e}")

print("\n" + "=" * 80)
print("PART 3: Numerical dE/dD for a specific descriptor")
print("=" * 80)

# Get descriptors
desc_0 = calc.get_descriptors(atoms, invariants_only=True)
print(f"Descriptors shape: {desc_0.shape}")

# Perturb position and see how descriptors AND energy change
atom_idx = 0
coord_idx = 2
pos_eps = 1e-6

atoms_pert = atoms.copy()
pos = atoms_pert.positions.copy()
pos[atom_idx, coord_idx] += pos_eps
atoms_pert.positions = pos

desc_1 = calc.get_descriptors(atoms_pert, invariants_only=True)
dD_dpos_numerical = (desc_1 - desc_0) / pos_eps  # Shape: (num_atoms, total_features)

atoms_pert.calc = calc
dE_dpos_numerical = -(atoms_pert.get_potential_energy() - atoms.get_potential_energy()) / pos_eps

print(f"\nNumerical dE/dpos[{atom_idx},{coord_idx}]: {dE_dpos_numerical:.6f}")
print(f"Numerical dD/dpos[0, :5] for atom {atom_idx}: {dD_dpos_numerical[0, :5]}")

# If dE/dpos = sum_i dE/dD[i] * dD[i]/dpos, then:
# dE/dD[i] ≈ dE/dpos / (dD[i]/dpos) for features where dD[i]/dpos is non-zero
# More precisely: sum_i dE/dD[i] * dD[i]/dpos = dE/dpos
# This is ONE equation with 256 unknowns, so we need more samples

print("\n" + "=" * 80)
print("PART 4: Use least squares to estimate dE/dD from multiple position perturbations")
print("=" * 80)

# Collect multiple samples: perturb different positions and record dD and dE
n_samples = 9  # 3 atoms * 3 coordinates
dD_matrix = []  # Each row: how ALL descriptors change for one position perturbation
dE_vector = []  # Each entry: how energy changes for one position perturbation

for atom_idx in range(len(atoms)):
    for coord_idx in range(3):
        atoms_pert = atoms.copy()
        atoms_pert.calc = calc
        pos = atoms_pert.positions.copy()
        pos[atom_idx, coord_idx] += pos_eps
        atoms_pert.positions = pos

        desc_pert = calc.get_descriptors(atoms_pert, invariants_only=True)
        E_pert = atoms_pert.get_potential_energy()

        # Change in descriptors (flattened)
        delta_D = (desc_pert - desc_0).flatten()  # All atoms, all features
        delta_E = E_pert - atoms.get_potential_energy()

        dD_matrix.append(delta_D / pos_eps)
        dE_vector.append(delta_E / pos_eps)

dD_matrix = np.array(dD_matrix)  # Shape: (9, 768) for 3 atoms * 256 features
dE_vector = np.array(dE_vector)  # Shape: (9,)

print(f"dD_matrix shape: {dD_matrix.shape}")
print(f"dE_vector shape: {dE_vector.shape}")

# Solve least squares: dE/dD such that dD_matrix @ dE_dD ≈ dE_vector
dE_dD_lstsq, residuals, rank, s = np.linalg.lstsq(dD_matrix, dE_vector, rcond=None)
print(f"\nLeast squares solution shape: {dE_dD_lstsq.shape}")
print(f"Residuals: {residuals}")
print(f"Rank: {rank}")

# Reshape to (num_atoms, num_features)
dE_dD_lstsq_reshaped = dE_dD_lstsq.reshape(len(atoms), -1)
print(f"Reshaped dE/dD: {dE_dD_lstsq_reshaped.shape}")

print("\n" + "=" * 80)
print("PART 5: Compare with get_energy_descriptors_gradients output")
print("=" * 80)

dE_dD_function = calc.get_energy_descriptors_gradients(atoms)
print(f"Function output shape: {dE_dD_function.shape}")

print(f"\nLeast squares dE/dD[0, :10]:\n{dE_dD_lstsq_reshaped[0, :10]}")
print(f"Function dE/dD[0, :10]:\n{dE_dD_function[0, :10]}")

print(f"\nDifference (first atom, first 10 features):")
print(dE_dD_lstsq_reshaped[0, :10] - dE_dD_function[0, :10])

print(f"\nMax absolute difference: {np.max(np.abs(dE_dD_lstsq_reshaped - dE_dD_function)):.6e}")
print(f"Relative difference: {np.linalg.norm(dE_dD_lstsq_reshaped - dE_dD_function) / np.linalg.norm(dE_dD_lstsq_reshaped):.6f}")
