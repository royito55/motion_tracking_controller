import onnx
import json

# Load the model
model = onnx.load('/tmp/wandb_onnx/2026-01-02_12-36-57_23dof_dance2_1.onnx')

# Check metadata
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")
    
# Check input/output shapes
print("\nInputs:")
for input in model.graph.input:
    print(f"  {input.name}: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

print("\nOutputs:")
for output in model.graph.output:
    print(f"  {output.name}: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
