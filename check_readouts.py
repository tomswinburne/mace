"""
Check if readouts are linear or nonlinear
"""
from mace.calculators import mace_mp

calc = mace_mp(model="small", device="cpu", default_dtype="float64")
model = calc.models[0]

print(f"Number of readouts: {len(model.readouts)}")
for i, readout in enumerate(model.readouts):
    print(f"\nReadout {i}:")
    print(f"  Type: {type(readout).__name__}")
    print(f"  Has 'linear': {hasattr(readout, 'linear')}")
    print(f"  Has 'linear_1': {hasattr(readout, 'linear_1')}")

    # Check if it's truly linear
    is_linear = hasattr(readout, 'linear') and not hasattr(readout, 'linear_1')
    print(f"  Is linear: {is_linear}")
