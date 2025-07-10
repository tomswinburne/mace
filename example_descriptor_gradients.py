#!/usr/bin/env python3
"""
Example script demonstrating how to compute derivatives of MACE descriptors 
with respect to atomic positions.
"""

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

def example_descriptor_gradients():
    """Example demonstrating descriptor gradient computation."""
    
    # Create a simple water molecule for testing
    atoms = Atoms(
        symbols=['H', 'H', 'O'],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5]
        ])
    )
    
    # Note: You'll need to provide a path to an actual MACE model
    # For this example, we assume you have a model file
    model_path = "path/to/your/mace_model.model"
    
    try:
        # Initialize MACE calculator
        calc = MACECalculator(model_paths=model_path, device="cpu")
        
        # Compute descriptors and their gradients
        descriptors, gradients = calc.get_descriptor_gradients(
            atoms=atoms,
            invariants_only=True,  # Only get rotationally invariant features
            num_layers=-1          # Use all layers
        )
        
        print(f"Descriptors shape: {descriptors.shape}")
        print(f"Gradients shape: {gradients.shape}")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Number of descriptor features: {descriptors.shape[1]}")
        
        # gradients[i, j, k, l] = d(descriptor[i,j])/d(position[k,l])
        # where:
        # i = atom index for the descriptor
        # j = feature index
        # k = atom index for the position
        # l = coordinate index (0=x, 1=y, 2=z)
        
        # Example: gradient of descriptor feature 0 of atom 0 w.r.t. x-position of atom 1
        grad_example = gradients[0, 0, 1, 0]
        print(f"∂(descriptor[0,0])/∂(x_position[1]) = {grad_example}")
        
        # Compute finite difference to verify gradient (optional validation)
        delta = 1e-6
        atoms_perturbed = atoms.copy()
        atoms_perturbed.positions[1, 0] += delta
        
        descriptors_perturbed, _ = calc.get_descriptor_gradients(
            atoms=atoms_perturbed,
            invariants_only=True,
            num_layers=-1
        )
        
        finite_diff = (descriptors_perturbed[0, 0] - descriptors[0, 0]) / delta
        print(f"Finite difference check: {finite_diff}")
        print(f"Relative error: {abs(grad_example - finite_diff) / abs(finite_diff):.6f}")
        
        return descriptors, gradients
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid MACE model file and update the model_path variable.")
        return None, None

def example_force_comparison():
    """
    Example showing how descriptor gradients relate to forces.
    Forces are gradients of energy w.r.t. positions, while descriptor gradients
    show how the atomic descriptors change with positions.
    """
    
    # Create a simple molecule
    atoms = Atoms(
        symbols=['H', 'H'],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.5]
        ])
    )
    
    model_path = "path/to/your/mace_model.model"
    
    try:
        calc = MACECalculator(model_paths=model_path, device="cpu")
        
        # Get regular forces (energy gradients)
        atoms.calc = calc
        forces = atoms.get_forces()
        print(f"Forces shape: {forces.shape}")
        print(f"Force on atom 0: {forces[0]}")
        
        # Get descriptor gradients
        descriptors, desc_gradients = calc.get_descriptor_gradients(atoms)
        print(f"Descriptor gradients shape: {desc_gradients.shape}")
        
        # The descriptor gradients show how each descriptor feature changes
        # with atomic positions, which can be useful for understanding
        # the local chemical environment sensitivity
        
        return forces, desc_gradients
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid MACE model file.")
        return None, None

if __name__ == "__main__":
    print("MACE Descriptor Gradients Example")
    print("=" * 40)
    
    # Note: Update the model_path in the functions above to point to your MACE model
    print("\n1. Basic descriptor gradients example:")
    descriptors, gradients = example_descriptor_gradients()
    
    print("\n2. Comparison with forces:")
    forces, desc_gradients = example_force_comparison()
    
    print("\nUsage Notes:")
    print("- Descriptors represent the local atomic environment")
    print("- Descriptor gradients show how these environments change with atomic positions")
    print("- This is useful for understanding chemical sensitivity and feature engineering")
    print("- The gradients tensor has shape (n_atoms, n_features, n_atoms, 3)")
    print("  where gradients[i,j,k,l] = ∂(descriptor[i,j])/∂(position[k,l])")
