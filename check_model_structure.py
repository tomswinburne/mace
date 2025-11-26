"""
Check what energy components the MACE model has
"""
from mace.calculators import mace_mp

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
model = calc.models[0]

print("Model type:", type(model).__name__)
print("Has pair_repulsion:", hasattr(model, 'pair_repulsion'))
print("Has atomic_energies_fn:", hasattr(model, 'atomic_energies_fn'))
print("Has joint_embedding:", hasattr(model, 'joint_embedding'))
print("Number of readouts:", len(model.readouts))
print("Number of interactions:", model.num_interactions)
