"""
Debug energy-descriptor gradients computation
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

# Get energy components
calc.calculate(atoms)
print("Total energy:", atoms.get_potential_energy())
if 'node_energy' in calc.results:
    print("Node energies:", calc.results['node_energy'])
    print("Sum of node energies:", calc.results['node_energy'].sum())

# Manually compute energy through model
model = calc.models[0]
batch = calc._atoms_to_batch(atoms)
with torch.no_grad():
    out = model(batch.to_dict(), compute_stress=False, training=False)

print("\nModel outputs:")
print("Energy from model:", out['energy'].item())
if 'contributions' in out:
    print("Energy contributions shape:", out['contributions'].shape)
    print("Energy contributions:", out['contributions'])
    print("Sum of contributions:", out['contributions'].sum().item())

# Check what dE/dD gives us
dE_dD = calc.get_energy_descriptors_gradients(atoms)
print("\ndE/dD shape:", dE_dD.shape)
print("dE/dD mean:", np.abs(dE_dD).mean())

# Numerical check: perturb descriptors somehow and see energy change
# This is tricky since we can't directly set descriptors...
# Instead, verify using finite differences on energy w.r.t. positions
eps = 1e-6
forces_numerical = np.zeros((len(atoms), 3))
E0 = atoms.get_potential_energy()

for i in range(len(atoms)):
    for j in range(3):
        atoms_pert = atoms.copy()
        atoms_pert.calc = calc
        pos = atoms_pert.positions.copy()
        pos[i,j] += eps
        atoms_pert.positions = pos
        E1 = atoms_pert.get_potential_energy()
        forces_numerical[i,j] = -(E1 - E0) / eps

print("\nNumerical forces:\n", forces_numerical)
print("ASE forces:\n", atoms.get_forces())
print("Difference:", np.max(np.abs(forces_numerical - atoms.get_forces())))
