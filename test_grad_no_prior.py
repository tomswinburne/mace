"""Test gradient computation WITHOUT calling get_energy_descriptors_gradients first"""
import numpy as np
import torch
from ase import Atoms
from mace.calculators import mace_mp
from e3nn import o3
from mace.modules.utils import extract_invariant

atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]
)

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
atoms.calc = calc

model = calc.models[0]
batch = calc._atoms_to_batch(atoms)

num_interactions = int(model.num_interactions)
irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
l_max = irreps_out.lmax
num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

print("Computing TWO gradients in sequence WITHOUT prior get_energy_descriptors_gradients call...")

for i in range(2):
    print(f"\nGradient computation {i+1}...")
    batch_dict = {}
    for key, value in batch.to_dict().items():
        if torch.is_tensor(value):
            batch_dict[key] = torch.tensor(value.detach().cpu().numpy(), dtype=value.dtype, device=calc.device)
        else:
            batch_dict[key] = value

    batch_dict['positions'].requires_grad_(True)

    with torch.enable_grad():
        out = model(batch_dict, compute_stress=False, training=True)
        node_feats_concat = out["node_feats"]

        descriptors_list = []
        for layer_idx in range(num_interactions):
            layer_feats = node_feats_concat[:, layer_idx * irreps_out.dim : (layer_idx + 1) * irreps_out.dim]
            layer_invariants = extract_invariant(
                layer_feats.unsqueeze(0),
                num_layers=1,
                num_features=num_invariant_features,
                l_max=l_max,
            ).squeeze(0)
            descriptors_list.append(layer_invariants)

        descriptors = torch.cat(descriptors_list, dim=1)
        scalar_output = descriptors[0, i]  # Different feature each time

        grad_output = torch.autograd.grad(scalar_output, batch_dict['positions'], create_graph=False)[0]
        print(f"SUCCESS! Gradient shape: {grad_output.shape}")

print("\n✓ Both gradients computed successfully!")
