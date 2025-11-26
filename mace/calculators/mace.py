###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging

# pylint: disable=wrong-import-position
import os
from glob import glob
from pathlib import Path
from typing import List, Union

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data as mace_data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model

try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    CUEQQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQQ_AVAILABLE = False
    run_e3nn_to_cueq = None

try:
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq

    OEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OEQ_AVAILABLE = False
    run_e3nn_to_oeq = None

try:
    import intel_extension_for_pytorch as ipex

    has_ipex = True
except ImportError:
    has_ipex = False


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu or xpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_paths: Union[list, str, None] = None,
        models: Union[List[torch.nn.Module], torch.nn.Module, None] = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        info_keys=None,
        arrays_keys=None,
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        enable_cueq=False,
        enable_oeq=False,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        if enable_cueq or enable_oeq:
            assert model_type == "MACE", "CuEq only supports MACE models"
            if compile_mode is not None:
                logging.warning(
                    "CuEq or Oeq does not support torch.compile, setting compile_mode to None"
                )
                compile_mode = None
        if enable_cueq and enable_oeq:
            raise ValueError(
                "CuEq and OEq cannot be used together, please choose one of them"
            )
        if enable_cueq and not CUEQQ_AVAILABLE:
            raise ImportError(
                "cuequivariance is not installed so CuEq acceleration cannot be used"
            )
        if enable_oeq and not OEQ_AVAILABLE:
            raise ImportError(
                "openequivariance is not installed so OEq acceleration cannot be used"
            )
        if "model_path" in kwargs:
            deprecation_message = (
                "'model_path' argument is deprecated, please use 'model_paths'"
            )
            if model_paths is None:
                logging.warning(f"{deprecation_message} in the future.")
                model_paths = kwargs["model_path"]
            else:
                raise ValueError(
                    f"both 'model_path' and 'model_paths' given, {deprecation_message} only."
                )

        if (model_paths is None) == (models is None):
            raise ValueError(
                "Exactly one of 'model_paths' or 'models' must be provided"
            )

        self.results = {}
        if info_keys is None:
            info_keys = {"total_spin": "spin", "total_charge": "charge"}
        if arrays_keys is None:
            arrays_keys = {}
        self.info_keys = info_keys
        self.arrays_keys = arrays_keys

        self.model_type = model_type
        self.compute_atomic_stresses = False

        if model_type not in [
            "MACE",
            "DipoleMACE",
            "EnergyDipoleMACE",
            "DipolePolarizabilityMACE",
        ]:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, DipolePolarizabilityMACE, EnergyDipoleMACE], {model_type} not supported"
            )

        # superclass constructor initializes self.implemented_properties to an empty list
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            self.implemented_properties.extend(
                [
                    "energy",
                    "energies",
                    "free_energy",
                    "node_energy",
                    "forces",
                    "stress",
                ]
            )
            if kwargs.get("compute_atomic_stresses", False):
                self.implemented_properties.extend(["stresses", "virials"])
                self.compute_atomic_stresses = True
        if model_type in ["EnergyDipoleMACE", "DipoleMACE", "DipolePolarizabilityMACE"]:
            self.implemented_properties.extend(["dipole"])
        if model_type == "DipolePolarizabilityMACE":
            self.implemented_properties.extend(
                [
                    "charges",
                    "polarizability",
                    "polarizability_sh",
                ]
            )

        if model_paths is not None:
            if isinstance(model_paths, str):
                # Find all models that satisfy the wildcard (e.g. mace_model_*.pt)
                model_paths_glob = glob(model_paths)

                if len(model_paths_glob) == 0:
                    raise ValueError(f"Couldn't find MACE model files: {model_paths}")

                model_paths = model_paths_glob
            elif isinstance(model_paths, Path):
                model_paths = [model_paths]

            if len(model_paths) == 0:
                raise ValueError("No mace file names supplied")
            self.num_models = len(model_paths)

            # Load models from files
            self.models = [
                torch.load(f=model_path, map_location=device)
                for model_path in model_paths
            ]

        elif models is not None:
            if not isinstance(models, list):
                models = [models]

            if len(models) == 0:
                raise ValueError("No models supplied")

            self.models = models
            self.num_models = len(models)

        if self.num_models > 1:
            logging.info(f"Running committee mace with {self.num_models} models")

            if model_type in ["MACE", "EnergyDipoleMACE"]:
                self.implemented_properties.extend(
                    ["energy_comm", "energy_var", "forces_comm", "stress_var"]
                )
            if model_type in [
                "DipoleMACE",
                "EnergyDipoleMACE",
                "DipolePolarizabilityMACE",
            ]:
                self.implemented_properties.extend(["dipole_var"])

        if compile_mode is not None:
            logging.info(f"Torch compile is enabled with mode: {compile_mode}")
            self.models = [
                torch.compile(
                    prepare(extract_model)(model=model, map_location=device),
                    mode=compile_mode,
                    fullgraph=fullgraph,
                )
                for model in self.models
            ]
            self.use_compile = True
        else:
            self.use_compile = False

        # Ensure all models are on the same device
        for model in self.models:
            model.to(device)

        if has_ipex and device == "xpu":
            for model in self.models:
                model = ipex.optimize(model)

        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"committee r_max are not all the same {' '.join(r_maxs)}")
        self.r_max = float(r_maxs[0])

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key

        try:
            self.available_heads: List[str] = self.models[0].heads  # type: ignore
        except AttributeError:
            self.available_heads = ["Default"]
        kwarg_head = kwargs.get("head", None)
        if kwarg_head is not None:
            self.head = kwarg_head
            if isinstance(self.head, str):
                if self.head not in self.available_heads:
                    last_head = self.available_heads[-1]
                    logging.warning(
                        f"Head {self.head} not found in available heads {self.available_heads}, defaulting to the last head: {last_head}"
                    )
                    self.head = last_head
        elif len(self.available_heads) == 1:
            self.head = self.available_heads[0]
        else:
            self.head = [
                head for head in self.available_heads if head.lower() == "default"
            ]
            if len(self.head) == 0:
                raise ValueError(
                    "Head keyword was not provided, and no head in the model is 'default'. "
                    "Please provide a head keyword to specify the head you want to use. "
                    f"Available heads are: {self.available_heads}"
                )
            self.head = self.head[0]

        logging.info(f"Using head {self.head} out of  {self.available_heads}")

        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            logging.warning(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            logging.warning(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)
        if enable_cueq:
            logging.info("Converting models to CuEq for acceleration")
            self.models = [
                run_e3nn_to_cueq(model, device=device).to(device)
                for model in self.models
            ]
        if enable_oeq:
            logging.info("Converting models to OEq for acceleration")
            self.models = [
                run_e3nn_to_oeq(model, device=device).to(device)
                for model in self.models
            ]
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def check_state(self, atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def _create_result_tensors(
        self, num_models: int, num_atoms: int, batch, out: dict
    ) -> dict:
        # unfortunately, code is expecting shape that isn't always same as underlying model
        # output tensor shape, e.g. stress is returned as 1x3x3 and we want 3x3
        tensor_shapes = {
            "energy": [],
            "node_energy": [num_atoms],
            "forces": [num_atoms, 3],
            "stress": [3, 3],
            "atomic_stresses": [num_atoms, 3, 3],
            "atomic_virials": [num_atoms, 3, 3],
            "dipole": [3],
            "charges": [num_atoms],
            "polarizability": [3, 3],
            "polarizability_sh": [6],
        }
        dict_of_tensors = {}
        for key in out:
            if key not in tensor_shapes or out.get(key) is None:
                continue
            shape = [num_models] + tensor_shapes[key]
            dict_of_tensors[key] = torch.zeros(*shape, device=self.device)

        node_e0 = None
        if "node_energy" in out:
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = (
                self.models[0]
                .atomic_energies_fn(batch["node_attrs"])[num_atoms_arange, node_heads]
                .detach()
                .cpu()
                .numpy()
            )

        return dict_of_tensors, node_e0

    def _atoms_to_batch(self, atoms):
        self.arrays_keys.update({self.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(
            info_keys=self.info_keys, arrays_keys=self.arrays_keys
        )
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_data.AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    heads=self.available_heads,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = None
        node_e0 = None
        # copy from output of model() call to ret_tensors
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=self.use_compile,
                compute_edge_forces=self.compute_atomic_stresses,
                compute_atomic_stresses=self.compute_atomic_stresses,
            )
            if i == 0:
                ret_tensors, node_e0 = self._create_result_tensors(
                    self.num_models, len(atoms), batch, out
                )
            for key, val in ret_tensors.items():
                if out.get(key) is not None:
                    val[i] = out[key].detach()

        # covert from ret_tensors to calculator results dict
        self.results = {}
        scalar_tensors = set(["energy"])
        results_store_ensemble = set(["energy", "forces", "stress", "dipole"])
        for results_key, ret_key, unit_conv in [
            ("energy", "energy", self.energy_units_to_eV),
            ("node_energy", "node_energy", self.energy_units_to_eV),
            ("forces", "forces", self.energy_units_to_eV / self.length_units_to_A),
            ("stress", "stress", self.energy_units_to_eV / self.length_units_to_A**3),
            (
                "stresses",
                "atomic_stresses",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            (
                "virials",
                "atomic_virials",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            ("dipole", "dipole", 1.0),
            ("charges", "charges", 1.0),
            ("polarizability", "polarizability", 1.0),
            ("polarizability_sh", "polarizability_sh", 1.0),
        ]:
            if ret_tensors.get(ret_key) is not None:
                data = torch.mean(ret_tensors[ret_key], dim=0).cpu()
                if ret_key in scalar_tensors:
                    data = data.item()
                else:
                    data = data.numpy()
                self.results[results_key] = data * unit_conv

                if self.num_models > 1 and results_key in results_store_ensemble:
                    data = ret_tensors[results_key].cpu().numpy()
                    data *= unit_conv
                    self.results[results_key + "_comm"] = data

                    data = torch.var(
                        ret_tensors[results_key], dim=0, unbiased=False
                    ).cpu()
                    if ret_key in scalar_tensors:
                        data = data.item()
                    else:
                        data = data.numpy()
                    data *= unit_conv
                    self.results[results_key + "_var"] = data

        # special cases
        if self.results.get("energy") is not None:
            self.results["free_energy"] = self.results["energy"]
        if self.results.get("node_energy") is not None:
            self.results["energies"] = self.results["node_energy"].copy()
            self.results["node_energy"] -= node_e0
        if self.results.get("stress") is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        if self.results.get("stresses") is not None:
            self.results["stresses"] = np.asarray(
                [
                    full_3x3_to_voigt_6_stress(stress)
                    for stress in self.results["stresses"]
                ]
            )

    def get_dielectric_derivatives(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type not in ["DipoleMACE", "DipolePolarizabilityMACE"]:
            raise NotImplementedError(
                "Only implemented for DipoleMACE or DipolePolarizabilityMACE models"
            )
        batch = self._atoms_to_batch(atoms)
        outputs = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_dielectric_derivatives=True,
                training=self.use_compile,
            )
            for model in self.models
        ]
        dipole_derivatives = [
            output["dmu_dr"].clone().detach().cpu().numpy() for output in outputs
        ]
        if self.models[0].use_polarizability:
            polarizability_derivatives = [
                output["dalpha_dr"].clone().detach().cpu().numpy() for output in outputs
            ]
            if self.num_models == 1:
                dipole_derivatives = dipole_derivatives[0]
                polarizability_derivatives = polarizability_derivatives[0]
            del outputs, batch, atoms
            return dipole_derivatives, polarizability_derivatives
        if self.num_models == 1:
            return dipole_derivatives[0]
        del outputs, batch, atoms
        return dipole_derivatives

    def get_hessian(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        hessians = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=self.use_compile,
            )["hessian"]
            for model in self.models
        ]
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray of shape (num_atoms, total_features) where total_features = num_layers * num_invariant_features, if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions
        batch = self._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        if invariants_only:
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = [
            descriptor[:, :to_keep].detach().cpu().numpy() for descriptor in descriptors
        ]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors

    def get_descriptors_gradients(
        self, atoms=None, weight_tensor=None, num_layers=-1, use_finite_diff=False
    ):
        """Extracts the spatial gradients of weighted descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param weight_tensor: np.ndarray of shape (num_axes, num_atoms), weights for each atom along different axes. If None, defaults to ones with shape (1, num_atoms).
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :param use_finite_diff: bool, if True use finite differences instead of autograd (slower but more robust)
        :return: np.ndarray of shape (num_axes, num_atoms, 3, total_features) where total_features = num_layers * num_invariant_features, containing spatial gradients of the weighted descriptor sum for each axis
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if self.num_models != 1:
            raise NotImplementedError(
                "Only implemented for single models (num_models=1)"
            )

        num_atoms = len(atoms)
        if weight_tensor is None:
            weight_tensor = np.ones((1, num_atoms))
        else:
            weight_tensor = np.asarray(weight_tensor)
            if weight_tensor.ndim == 1:
                # Backward compatibility: if 1D array provided, reshape to (1, num_atoms)
                weight_tensor = weight_tensor.reshape(1, -1)
            if weight_tensor.ndim != 2 or weight_tensor.shape[1] != num_atoms:
                raise ValueError(
                    f"weight_tensor must have shape (num_axes, {num_atoms}), got {weight_tensor.shape}"
                )

        num_axes = weight_tensor.shape[0]

        # Get model metadata to understand descriptor structure
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

        # Initialize gradient storage
        total_features = num_layers * num_invariant_features
        gradients = np.zeros((num_axes, num_atoms, 3, total_features))

        if use_finite_diff:
            # Fallback to finite differences if autograd fails
            # Get descriptors at current position - shape: (num_atoms, total_features)
            descriptors_0_flat = self.get_descriptors(atoms, invariants_only=True, num_layers=num_layers)
            descriptors_0 = descriptors_0_flat.reshape(num_atoms, num_layers, num_invariant_features)
            weighted_desc_0 = np.einsum('ai,ijk->ajk', weight_tensor, descriptors_0)

            eps = 1e-6
            for atom_idx in range(num_atoms):
                for coord_idx in range(3):
                    atoms_pert = atoms.copy()
                    pos = atoms_pert.positions.copy()
                    pos[atom_idx, coord_idx] += eps
                    atoms_pert.positions = pos

                    descriptors_pert_flat = self.get_descriptors(
                        atoms_pert, invariants_only=True, num_layers=num_layers
                    )
                    descriptors_pert = descriptors_pert_flat.reshape(num_atoms, num_layers, num_invariant_features)
                    weighted_desc_pert = np.einsum('ai,ijk->ajk', weight_tensor, descriptors_pert)

                    for axis_idx in range(num_axes):
                        grad_2d = (weighted_desc_pert[axis_idx] - weighted_desc_0[axis_idx]) / eps
                        gradients[axis_idx, atom_idx, coord_idx, :] = grad_2d.flatten()
        else:
            # Use autograd through the model
            import torch

            model = self.models[0]
            batch = self._atoms_to_batch(atoms)

            # Convert weight_tensor to torch
            weight_tensor_torch = torch.tensor(
                weight_tensor, dtype=torch.get_default_dtype(), device=self.device
            )

            # For each axis and feature, compute gradients
            # Note: We need a fresh forward pass for each feature to avoid graph issues
            model.eval()  # Ensure model is in eval mode
            model.zero_grad()  # Clear any lingering gradients

            for axis_idx in range(num_axes):
                weights = weight_tensor_torch[axis_idx]  # shape: (num_atoms,)

                for feat_idx in range(total_features):
                    # Create completely fresh batch for each computation to avoid graph reuse
                    # Convert everything to numpy and back to ensure NO shared computation graphs
                    fresh_batch = self._atoms_to_batch(atoms)
                    fresh_batch_dict = {}
                    for key, value in fresh_batch.to_dict().items():
                        if torch.is_tensor(value):
                            # Go through numpy to completely break any graph connections
                            numpy_data = value.detach().cpu().numpy()
                            fresh_batch_dict[key] = torch.tensor(
                                numpy_data,
                                dtype=value.dtype,
                                device=self.device
                            )
                        else:
                            fresh_batch_dict[key] = value

                    # Now set requires_grad on positions
                    fresh_batch_dict['positions'].requires_grad_(True)
                    positions = fresh_batch_dict['positions']

                    with torch.enable_grad():
                        # Forward pass - IMPORTANT: use training=True to avoid graph conflicts
                        # When training=False, the model internally calls requires_grad_() which conflicts
                        out = model(fresh_batch_dict, compute_stress=False, training=True)
                        node_feats_concat = out["node_feats"]

                        # Extract invariant descriptors for each layer
                        descriptors_list = []
                        for layer_idx in range(num_layers):
                            layer_feats = node_feats_concat[:, layer_idx * irreps_out.dim : (layer_idx + 1) * irreps_out.dim]
                            layer_invariants = extract_invariant(
                                layer_feats.unsqueeze(0),
                                num_layers=1,
                                num_features=num_invariant_features,
                                l_max=l_max,
                            ).squeeze(0)
                            descriptors_list.append(layer_invariants)

                        # Concatenate: (num_atoms, total_features)
                        descriptors = torch.cat(descriptors_list, dim=1)

                        # Apply weights: sum over atoms -> (total_features,)
                        weighted_descriptors = (weights.unsqueeze(1) * descriptors).sum(dim=0)
                        scalar_output = weighted_descriptors[feat_idx]

                        # Compute gradient
                        grad_output = torch.autograd.grad(
                            scalar_output,
                            positions,
                            create_graph=False,
                        )[0]  # (num_atoms, 3)

                        gradients[axis_idx, :, :, feat_idx] = grad_output.detach().cpu().numpy()

                    # Clean up
                    del positions, out, node_feats_concat, descriptors_list, descriptors, weighted_descriptors, scalar_output, grad_output
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return gradients

    def get_energy_descriptors_gradients(self, atoms=None, num_layers=-1):
        """Computes the gradient of energy with respect to descriptors.

        This function computes dE/dD where E is the total energy and D are the
        invariant descriptors at each layer. For linear readout heads, this is
        simply the readout weights. For nonlinear readouts, autograd is used.

        :param atoms: ase.Atoms object
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray of shape (num_atoms, total_features) where total_features = num_layers * num_invariant_features
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if self.num_models != 1:
            raise NotImplementedError(
                "Only implemented for single models (num_models=1)"
            )

        # Get model metadata
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

        num_atoms = len(atoms)
        total_features = num_layers * num_invariant_features

        model = self.models[0]

        # Check if all readouts are linear
        all_linear = all(hasattr(readout, 'linear') and not hasattr(readout, 'linear_1')
                        for readout in model.readouts[:num_layers])

        # Get scale factors from scale_shift block
        scale_shift = model.scale_shift
        if hasattr(scale_shift, 'scale'):
            scales = scale_shift.scale.detach().cpu().numpy()
        else:
            scales = np.ones(1)

        if all_linear:
            # Analytical gradient: for linear readouts, dE/dD is just the readout weights
            # times the scale factor from scale_shift
            # For E = scale * (sum_i sum_j w_j * D_ij) + shift, we have dE/dD_ij = scale * w_j
            gradients = np.zeros((num_atoms, total_features))

            # Get node_heads to determine which scale to use for each atom
            batch = self._atoms_to_batch(atoms)
            with torch.no_grad():
                out_base = model(batch.to_dict(), compute_stress=False, training=False)

            if 'node_heads' in out_base:
                node_heads = out_base['node_heads'].cpu().numpy()
            elif hasattr(batch, 'heads'):
                node_heads = batch.heads.cpu().numpy()
            else:
                node_heads = np.zeros(num_atoms, dtype=np.int64)

            for layer_idx in range(num_layers):
                if layer_idx < len(model.readouts):
                    readout = model.readouts[layer_idx]
                    # Get the linear weight: shape is (num_heads, num_features_in)
                    weight = readout.linear.weight.detach().cpu().numpy()

                    start_idx = layer_idx * num_invariant_features
                    end_idx = (layer_idx + 1) * num_invariant_features

                    # The readout operates on invariant features only
                    # weight shape could be (1, num_invariant_features) or similar
                    # We need to extract the relevant part that operates on invariants

                    if weight.shape[1] == num_invariant_features:
                        # Direct match - weight operates on invariant features
                        # Apply readout weights and scale factor per atom
                        for atom_idx in range(num_atoms):
                            head = node_heads[atom_idx]
                            scale = scales[head] if len(scales) > 1 else scales[0]
                            gradients[atom_idx, start_idx:end_idx] = scale * weight[head, :]
                    else:
                        # Weight operates on full irreps - need to extract invariant part
                        for atom_idx in range(num_atoms):
                            head = node_heads[atom_idx]
                            scale = scales[head] if len(scales) > 1 else scales[0]
                            gradients[atom_idx, start_idx:end_idx] = scale * weight[head, :num_invariant_features]

            return gradients

        else:
            # Nonlinear readout: use autograd through the readout heads
            # We need to:
            # 1. Get the descriptors (node_feats) from the model
            # 2. Enable gradients on them
            # 3. Pass through readout heads
            # 4. Compute gradients w.r.t. descriptors

            batch = self._atoms_to_batch(atoms)

            gradients = np.zeros((num_atoms, total_features))

            # Run forward pass and extract node_feats
            with torch.enable_grad():
                out = model(batch.to_dict(), compute_stress=False, training=False)

                # node_feats_out is concatenated features from all layers
                # Shape: (num_atoms, num_layers * full_irreps_dim)
                node_feats_concat = out["node_feats"]

                # Extract invariant descriptors for each layer
                # We need to manually extract invariants and enable gradients
                descriptors_list = []
                for i in range(num_layers):
                    # Get this layer's node_feats
                    layer_feats = node_feats_concat[:, i * irreps_out.dim : (i + 1) * irreps_out.dim]

                    # Extract invariant features
                    layer_invariants = extract_invariant(
                        layer_feats.unsqueeze(0),
                        num_layers=1,
                        num_features=num_invariant_features,
                        l_max=l_max,
                    ).squeeze(0)  # Shape: (num_atoms, num_invariant_features)

                    descriptors_list.append(layer_invariants)

                # Concatenate: shape (num_atoms, total_features)
                descriptors = torch.cat(descriptors_list, dim=1)
                descriptors = descriptors.detach().clone().requires_grad_(True)

                # Now we need to recompute energy from these descriptors
                # Problem: we can't easily do this without rerunning the readout heads
                # on the *full* node_feats (not just invariants)

                # Alternative approach: compute jacobian using torch.autograd.functional
                # For each layer, we need dE/d(layer_invariants)

                # This is complex - let's use a simpler numerical approach
                # For each descriptor feature, compute dE/dD numerically

            # Numerical differentiation approach
            # Get baseline descriptors and energy
            descriptors_0 = self.get_descriptors(atoms, invariants_only=True, num_layers=num_layers)
            descriptors_0 = descriptors_0.reshape(num_atoms, num_layers, num_invariant_features)

            E_0 = atoms.copy()
            E_0.calc = self
            energy_0 = E_0.get_potential_energy()

            # For each descriptor, we need to find how to perturb it via positions
            # This is the inverse problem and is expensive
            # Better: use finite differences in energy/position space combined with
            # descriptor gradients

            # Using dE/dD = sum_ia (dE/dpos_ia) * (dpos_ia/dD)
            # But (dpos/dD) is hard to compute

            # Simpler: use the fact that for small perturbations,
            # delta_E ≈ sum_ij (dE/dD_ij) * delta_D_ij
            # We can solve this as a least-squares problem by perturbing positions

            # For now, let's use a different approach: compute gradients layer by layer
            # using the fact that the readout is applied per-layer

            # The key insight: the readout operates on node_feats which contain
            # both invariant and non-invariant features. For equivariant features (l>0),
            # the readout must preserve equivariance, meaning only invariant (l=0)
            # features contribute to the scalar energy output.

            # Therefore, dE/d(non-invariant features) = 0, and we only need
            # dE/d(invariant features)

            for layer_idx in range(num_layers):
                readout = model.readouts[layer_idx]

                # Get this layer's descriptors with gradients enabled
                with torch.enable_grad():
                    # Re-run model to get fresh gradients
                    out_fresh = model(batch.to_dict(), compute_stress=False, training=False)
                    node_feats_concat_fresh = out_fresh["node_feats"]

                    # Extract this layer's features
                    layer_feats = node_feats_concat_fresh[:, layer_idx * irreps_out.dim : (layer_idx + 1) * irreps_out.dim]

                    # The layer_feats contain both invariant and equivariant features
                    # arranged according to the irreps structure
                    # For l=0,1,2,...,l_max, we have (2l+1) components for each l
                    # The first num_invariant_features are the l=0 (invariant) features

                    # Extract only invariant features
                    layer_invariants = extract_invariant(
                        layer_feats.unsqueeze(0),
                        num_layers=1,
                        num_features=num_invariant_features,
                        l_max=l_max,
                    ).squeeze(0)  # Shape: (num_atoms, num_invariant_features)
                    layer_invariants = layer_invariants.clone().detach().requires_grad_(True)

                    # For the readout, we need to reconstruct full node_feats with only
                    # invariants having gradients. But this is complex.
                    # Better: directly compute gradient by finite differences on invariants

                # Finite difference approach: perturb each invariant feature slightly
                # and measure energy change
                eps = 1e-5
                for feat_idx in range(num_invariant_features):
                    # We need to perturb the invariant feature and recompute energy
                    # But we can't directly modify descriptors in the model
                    # This approach won't work either

                    # Alternative: since we have the readout weights/function,
                    # we can compute the gradient analytically for each atom

                    pass  # Continue to next approach

            # New approach: For each atom individually, compute how its descriptor
            # gradients affect energy via the readout
            # For atom i: E_i = readout(descriptors_i)
            # So dE/dD_i = d(readout)/dD_i

            # Get node_heads
            batch = self._atoms_to_batch(atoms)

            # Check if heads are in batch
            if hasattr(batch, 'heads'):
                node_heads = batch.heads
            else:
                # Try to get from a forward pass (but we need to be careful about gradients)
                with torch.no_grad():
                    try:
                        out_base = model(batch.to_dict(), compute_stress=False, training=False)
                        if 'node_heads' in out_base:
                            node_heads = out_base['node_heads']
                        else:
                            node_heads = torch.zeros(num_atoms, dtype=torch.long, device=self.device)
                    except:
                        # If model call fails, assume single head
                        node_heads = torch.zeros(num_atoms, dtype=torch.long, device=self.device)

            # Get scale factors - these are applied to the SUM of all readout energies
            scale_shift = model.scale_shift
            if hasattr(scale_shift, 'scale'):
                scales = scale_shift.scale.detach().cpu().numpy()
            else:
                scales = np.ones(1)

            for layer_idx in range(num_layers):
                readout = model.readouts[layer_idx]

                # We'll compute the jacobian of the readout for each atom
                with torch.enable_grad():
                    # Get the descriptors for this layer
                    # IMPORTANT: use training=True to avoid graph conflicts
                    out_fresh = model(batch.to_dict(), compute_stress=False, training=True)
                    node_feats_concat = out_fresh["node_feats"]
                    layer_feats = node_feats_concat[:, layer_idx * irreps_out.dim : (layer_idx + 1) * irreps_out.dim]

                    # For each atom, compute gradient of its energy w.r.t. its descriptors
                    for atom_idx in range(num_atoms):
                        # Create input for just this atom with gradients
                        atom_feats = layer_feats[atom_idx:atom_idx+1].clone().detach().requires_grad_(True)
                        atom_head = node_heads[atom_idx:atom_idx+1]

                        # Apply readout (without scale_shift - that's applied to the sum of all readouts)
                        atom_energy = readout(atom_feats, atom_head)
                        if atom_energy.dim() > 1:
                            atom_energy = atom_energy[0, atom_head[0]]
                        else:
                            atom_energy = atom_energy[0]

                        # Compute gradient w.r.t. features
                        grad_feats = torch.autograd.grad(
                            atom_energy,
                            atom_feats,
                            create_graph=False,
                        )[0]  # Shape: (1, full_irreps_dim)

                        # Extract invariant part
                        grad_invariants = extract_invariant(
                            grad_feats,
                            num_layers=1,
                            num_features=num_invariant_features,
                            l_max=l_max,
                        ).squeeze(0)  # Shape: (num_invariant_features,)

                        # Apply scale factor (scale_shift is applied to SUM of all readouts)
                        head_val = atom_head[0].item()
                        if scales.ndim > 0 and scales.shape[0] > 1:
                            scale = scales[head_val]
                        else:
                            scale = scales.item() if scales.ndim == 0 else scales[0]

                        # Store
                        start_idx = layer_idx * num_invariant_features
                        end_idx = (layer_idx + 1) * num_invariant_features
                        gradients[atom_idx, start_idx:end_idx] = (scale * grad_invariants).detach().cpu().numpy()

            return gradients
