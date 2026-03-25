import json
import os
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

# Safely resolve paths relative to where this script is located (the data/ folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "feynman_parse_trees.json")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "data_clouds.json")

# Parameters
N_SAMPLES = np.random.randint(30, 201)  # number of datapoints per equation

print(f"Loading data from local file: {INPUT_JSON}")
try:
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_JSON}. Please ensure you have cloned the repository correctly.")
    exit()

output_data = []

def evaluate_formula(expr_str, input_vars, var_bounds, constants, n=N_SAMPLES):
    # Define sympy symbols for each variable
    symbols = {var: sympy.Symbol(var) for var in input_vars}
    local_dict = {**symbols, **constants}

    # Parse the formula
    try:
        expr = parse_expr(expr_str, local_dict=local_dict)
        func = sympy.lambdify([symbols[v] for v in input_vars], expr, modules=["numpy"])
    except Exception as e:
        print(f"Error parsing expression: {expr_str} -> {e}")
        return None

    # Sample input values
    inputs = []
    for var in input_vars:
        low, high = var_bounds[var]
        samples = np.random.uniform(low, high, size=n)
        inputs.append(samples)

    inputs = np.array(inputs)  # shape: (d, n)

    try:
        outputs = func(*inputs)
        outputs = np.array(outputs)

        # Handle scalar output (e.g., if the equation evaluates to a constant)
        if outputs.ndim == 0:
            outputs = np.full(n, outputs)
        elif outputs.shape != (n,):
            return None  # Skip if not 1D output

        data_matrix = np.column_stack((inputs.T, outputs))  # shape: (n, d+1)
        return data_matrix
    except Exception as e:
        print(f"Evaluation error: {expr_str} -> {e}")
        return None

# Process each entry
print("Generating data clouds...")
for row in tqdm(data):
    row_id = row["row"]
    expr_str = row["original_formula"]

    # Parse input variables and bounds
    var_bounds = {}
    input_vars = []
    for var, meta in row["variables"].items():
        if meta["type"] == "variable":
            var_bounds[var] = (meta["low"], meta["high"])
            input_vars.append(var)

    # Parse constants safely (handles both dicts with 'value' and raw floats)
    constants_raw = row.get("constants", {})
    constants = {
        k: float(v.get("value", v)) if isinstance(v, dict) else float(v) 
        for k, v in constants_raw.items()
    }

    # Evaluate equation
    result = evaluate_formula(expr_str, input_vars, var_bounds, constants)

    if result is not None:
        output_data.append({
            "row_id": row_id,
            "inputs": input_vars,
            "data": result.tolist()
        })

# Save output
print(f"Saving output to local file: {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Done. Generated {len(output_data)} data clouds in the data/ directory.")
