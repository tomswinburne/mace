"""Test if even a SINGLE gradient computation works"""
import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]
)

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
atoms.calc = calc

print("Testing single feature gradient computation...")

# Try just computing gradient for ONE feature
weight_tensor = np.array([[1.0, 0.0, 0.0]])  # Just atom 0

try:
    # This should work - just 1 axis, and we'll manually limit to 1 feature
    import torch
    from e3nn import o3
    from mace.modules.utils import extract_invariant

    model = calc.models[0]
    batch = calc._atoms_to_batch(atoms)

    num_interactions = int(model.num_interactions)
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
    total_features = num_interactions * num_invariant_features

    print(f"Total features: {total_features}")
    print("Computing gradient for feature 0...")

    # Fresh batch
    batch_dict = {}
    for key, value in batch.to_dict().items():
        if torch.is_tensor(value):
            numpy_data = value.detach().cpu().numpy()
            batch_dict[key] = torch.tensor(numpy_data, dtype=value.dtype, device=calc.device)
        else:
            batch_dict[key] = value

    batch_dict['positions'].requires_grad_(True)
    positions = batch_dict['positions']

    with torch.enable_grad():
        out = model(batch_dict, compute_stress=False, training=False)
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
        weighted_descriptors = descriptors[0, :]  # Just atom 0
        scalar_output = weighted_descriptors[0]  # Just feature 0

        grad_output = torch.autograd.grad(scalar_output, positions, create_graph=False)[0]
        print(f"Gradient shape: {grad_output.shape}")
        print(f"Gradient:\n{grad_output.detach().cpu().numpy()}")
        print("SUCCESS: First gradient computed!")

    # Now try a SECOND gradient with completely fresh setup
    print("\nComputing gradient for feature 1...")
    batch_dict_2 = {}
    for key, value in batch.to_dict().items():
        if torch.is_tensor(value):
            numpy_data = value.detach().cpu().numpy()
            batch_dict_2[key] = torch.tensor(numpy_data, dtype=value.dtype, device=calc.device)
        else:
            batch_dict_2[key] = value

    batch_dict_2['positions'].requires_grad_(True)
    positions_2 = batch_dict_2['positions']

    with torch.enable_grad():
        out_2 = model(batch_dict_2, compute_stress=False, training=False)
        node_feats_concat_2 = out_2["node_feats"]

        descriptors_list_2 = []
        for layer_idx in range(num_interactions):
            layer_feats_2 = node_feats_concat_2[:, layer_idx * irreps_out.dim : (layer_idx + 1) * irreps_out.dim]
            layer_invariants_2 = extract_invariant(
                layer_feats_2.unsqueeze(0),
                num_layers=1,
                num_features=num_invariant_features,
                l_max=l_max,
            ).squeeze(0)
            descriptors_list_2.append(layer_invariants_2)

        descriptors_2 = torch.cat(descriptors_list_2, dim=1)
        weighted_descriptors_2 = descriptors_2[0, :]  # Just atom 0
        scalar_output_2 = weighted_descriptors_2[1]  # Feature 1 this time

        grad_output_2 = torch.autograd.grad(scalar_output_2, positions_2, create_graph=False)[0]
        print(f"Gradient shape: {grad_output_2.shape}")
        print(f"Gradient:\n{grad_output_2.detach().cpu().numpy()}")
        print("SUCCESS: Second gradient computed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
