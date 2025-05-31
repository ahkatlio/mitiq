---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6 
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} pt, ddd, rem, zne, advanced, pipeline, tutorial
```

# Advanced Error Mitigation Pipeline: Combining PT, DDD, REM, and ZNE

Error mitigation techniques in quantum computing often address specific types of noise. In real quantum devices, multiple noise sources are present simultaneously, making it beneficial to combine different error mitigation strategies. This tutorial demonstrates how to build an advanced error mitigation pipeline by combining:

1.  **Pauli Twirling (PT)**: Converts coherent noise into stochastic Pauli noise.
2.  **Digital Dynamical Decoupling (DDD)**: Mitigates time-correlated noise by inserting decoupling sequences.
3.  **Readout Error Mitigation (REM)**: Corrects errors that occur during the measurement process.
4.  **Zero-Noise Extrapolation (ZNE)**: Estimates the ideal, noiseless expectation value by running experiments at different noise levels and extrapolating to the zero-noise limit.

We'll implement a step-by-step approach, analyze the impact of each technique individually, and demonstrate how to effectively combine them for maximum error reduction.

## Installation

First, let's ensure Mitiq is installed. If you're running this in an environment where Mitiq isn't pre-installed, the following cell will install it.

```{code-cell} ipython3
try:
    import mitiq
    print(f"Mitiq version {mitiq.__version__} is already installed.")
except (ImportError, ModuleNotFoundError): # Adjusted for broader compatibility
    print("Mitiq not found. Installing...")
    %pip install mitiq --quiet
    import mitiq
    print(f"Mitiq version {mitiq.__version__} installed.")
```

## Setup

Let's begin by importing the necessary libraries and modules.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import cirq
from functools import partial

# Mitiq imports
import mitiq
from mitiq import MeasurementResult, Observable, PauliString
from mitiq.benchmarks import generate_ghz_circuit
from mitiq import zne
from mitiq import rem
from mitiq import ddd
from mitiq.pt import generate_pauli_twirl_variants 
from mitiq.zne.inference import LinearFactory, RichardsonFactory, ExpFactory, PolyFactory 
from mitiq.ddd import rules
```

## Circuit and Observable

For this tutorial, we'll use a GHZ (Greenberger–Horne–Zeilinger) state preparation circuit. GHZ states are highly entangled and are often used in benchmarking quantum hardware due to their sensitivity to noise.

```{code-cell} ipython3
# Create a 4-qubit GHZ circuit
num_qubits = 4
circuit = generate_ghz_circuit(n_qubits=num_qubits)
print("GHZ Circuit:")
print(circuit)

# The observable we'll measure is the X⊗X⊗X⊗X operator.
# For an ideal GHZ state |Ψ⟩ = (|0000⟩ + |1111⟩)/√2, the expectation value ⟨Ψ|X⊗X⊗X⊗X|Ψ⟩ is 1.
obs = Observable(PauliString("X" * num_qubits))
print(f"\nObservable: {obs}")
```

## Comprehensive Noise Model

To demonstrate the benefits of each mitigation technique, we need a noise model that incorporates various error sources. Our `execute_with_noise` function will simulate:
*   **Coherent phase errors (RZ rotations)**: Targeted by Pauli Twirling.
*   **Time-correlated idle noise (modeled as Z rotations proportional to moment index)**: Targeted by Digital Dynamical Decoupling.
*   **Readout errors (bit-flips before measurement)**: Targeted by Readout Error Mitigation.
*   **Depolarizing noise**: A general stochastic noise, mitigated by Zero-Noise Extrapolation.

```{code-cell} ipython3
def execute_with_noise(
    circuit_to_run: cirq.Circuit,
    noise_level_param: float = 1.0,  # General scaling for ZNE
    rz_angle_param: float = 0.05,    # Coherent phase error (for PT)
    idle_error_param: float = 0.02,  # Time-correlated noise (for DDD)
    p_readout_param: float = 0.03,   # Readout error probability (for REM)
    depol_prob_param: float = 0.01,  # Depolarizing noise (for ZNE)
    repetitions: int = 4000 # Number of shots
) -> MeasurementResult:
    """
    Executes a circuit with a comprehensive noise model.
    The noise_level_param is primarily for ZNE to scale the overall noise impact.
    """
    # Make a copy of the circuit to avoid modifying the input
    noisy_circuit = circuit_to_run.copy()
    qubits = sorted(noisy_circuit.all_qubits())

    # Apply noise that scales with noise_level_param for ZNE purposes
    current_rz_angle = rz_angle_param * noise_level_param
    current_idle_error = idle_error_param * noise_level_param
    current_p_readout = p_readout_param * noise_level_param
    current_depol_prob = depol_prob_param * noise_level_param
    
    # Ensure probabilities are capped at 1.0
    current_p_readout = min(current_p_readout, 1.0)
    current_depol_prob = min(current_depol_prob, 1.0)


    # Intermediate circuit for moment-based noise insertion
    temp_circuit_ops = []
    
    for moment_idx, moment in enumerate(noisy_circuit.moments):
        temp_circuit_ops.append(moment)
        # 1. Coherent phase errors (RZ rotations) - targeted by PT
        if current_rz_angle > 0:
            temp_circuit_ops.append(cirq.Moment(cirq.rz(rads=current_rz_angle).on(q) for q in qubits))
            
        # 2. Time-correlated idle noise - targeted by DDD
        # Applied per qubit, scaling with moment index to simulate accumulation
        if current_idle_error > 0:
             temp_circuit_ops.append(cirq.Moment(cirq.Z(q)**( (moment_idx + 1) * current_idle_error / len(noisy_circuit.moments) ) for q in qubits))

    noisy_circuit_with_moment_noise = cirq.Circuit(temp_circuit_ops)
    
    # 3. Depolarizing noise - benefits from ZNE
    if current_depol_prob > 0:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise.with_noise(cirq.depolarize(p=current_depol_prob))
    else:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise
    
    # 4. Readout errors - targeted by REM
    if current_p_readout > 0:
        noisy_circuit_with_depol.append(cirq.bit_flip(p=current_p_readout).on_each(*qubits))
    
    # Add measurements to the circuit
    noisy_circuit_with_depol.append(cirq.measure(*qubits, key='m'))
    
    simulator = cirq.DensityMatrixSimulator() # Using DensityMatrixSimulator for noise modeling
    result = simulator.run(noisy_circuit_with_depol, repetitions=repetitions)
    
    bitstrings = result.measurements['m']
    
    return MeasurementResult(bitstrings)
```

## Establishing Baselines

First, let's determine the ideal (noiseless) expectation value and the unmitigated noisy expectation value.

```{code-cell} ipython3
# Define a noiseless execution function
noiseless_exec = partial(execute_with_noise, 
                         noise_level_param=0.0) # Setting noise_level_param to 0 effectively turns off scaled noises

ideal_result_val = obs.expectation(circuit, noiseless_exec).real
print(f"Ideal expectation value: {ideal_result_val:.6f}")

# These are the actual noise strengths we want to mitigate
base_rz_angle = 0.05
base_idle_error = 0.03
base_p_readout = 0.04
base_depol_prob = 0.02

noisy_exec = partial(execute_with_noise,
                     rz_angle_param=base_rz_angle,
                     idle_error_param=base_idle_error,
                     p_readout_param=base_p_readout,
                     depol_prob_param=base_depol_prob) 

noisy_result_val = obs.expectation(circuit, noisy_exec).real
print(f"Unmitigated noisy expectation value: {noisy_result_val:.6f}")
print(f"Initial absolute error: {abs(ideal_result_val - noisy_result_val):.6f}")
```

## Applying Individual Error Mitigation Techniques

Now, let's apply each technique individually to observe its impact. The `noisy_exec` defined above (with `noise_level_param=1.0`) will be used as the baseline noisy executor for these individual tests.

### 1. Pauli Twirling (PT)

Pauli Twirling aims to convert coherent noise into stochastic Pauli noise, which can sometimes be easier for other mitigation techniques (like ZNE) to handle or might average out to a less detrimental effect.

```{code-cell} ipython3
num_twirled_variants = 3 # Number of twirled circuits to average over
twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=num_twirled_variants, random_state=0) # Added random_state for reproducibility

pt_expectations = []
for tw_circuit_idx, tw_circuit in enumerate(twirled_circuits):
    print(f"Executing PT variant {tw_circuit_idx+1}/{num_twirled_variants}") # Optional: for progress tracking
    exp_val = obs.expectation(tw_circuit, noisy_exec).real # noisy_exec already has base noise levels
    pt_expectations.append(exp_val)

pt_result_val = np.mean(pt_expectations)
print(f"PT mitigated expectation value: {pt_result_val:.6f}")
print(f"Absolute error after PT: {abs(ideal_result_val - pt_result_val):.6f}")
```

### 2. Digital Dynamical Decoupling (DDD)

DDD inserts sequences of pulses (often identity operations in logical effect) to decouple qubits from certain types of environmental noise, particularly time-correlated or low-frequency noise.

```{code-cell} ipython3
ddd_rule = rules.xyxy 
ddd_circuit = ddd.insert_ddd_sequences(circuit, ddd_rule)
print("\nDDD Circuit:") # Optional: to view the modified circuit
print(ddd_circuit)

ddd_result_val = obs.expectation(ddd_circuit, noisy_exec).real
print(f"DDD mitigated expectation value: {ddd_result_val:.6f}")
print(f"Absolute error after DDD: {abs(ideal_result_val - ddd_result_val):.6f}")
```

### 3. Readout Error Mitigation (REM)

REM corrects errors that occur during the measurement process. We'll use the confusion matrix approach, assuming we know the probabilities of `0->1` and `1->0` readout errors.

```{code-cell} ipython3
# p0 is P(1|0), p1 is P(0|1)
# These should match the p_readout_param in noisy_exec for an accurate correction
p0_rem = base_p_readout 
p1_rem = base_p_readout 
inverse_confusion_matrix = rem.generate_inverse_confusion_matrix(
    num_qubits, p0=p0_rem, p1=p1_rem
)

rem_executor = rem.mitigate_executor(noisy_exec, inverse_confusion_matrix=inverse_confusion_matrix)

rem_result_val = obs.expectation(circuit, rem_executor).real
print(f"REM mitigated expectation value: {rem_result_val:.6f}")
print(f"Absolute error after REM: {abs(ideal_result_val - rem_result_val):.6f}")
```

### 4. Zero-Noise Extrapolation (ZNE)

ZNE runs the circuit at different amplified noise levels and extrapolates the results back to the zero-noise limit.

```{code-cell} ipython3
# For ZNE, the 'executor' it uses should be one that can accept a 'noise_level'
# Our execute_with_noise is designed for this with its 'noise_level_param'.
# We fix the base noise characteristics and let ZNE vary the 'noise_level_param'.
zne_base_exec_for_standalone_zne = partial(execute_with_noise,
                                           rz_angle_param=base_rz_angle,
                                           idle_error_param=base_idle_error,
                                           p_readout_param=base_p_readout,
                                           depol_prob_param=base_depol_prob)

scale_factors_zne = [1.0, 1.5, 2.0] # Example scale factors
zne_factory_standalone = LinearFactory(scale_factors=scale_factors_zne)

# The scale_noise function tells ZNE how to modify circuits or instruct the executor.
# Here, fold_global modifies the circuit. The executor then runs this modified circuit
# with its noise_level_param (which ZNE will pass based on scale_factors).
# If scale_noise was a function that expected the executor to handle scaling,
# then ZNE would pass the scale_factor to the executor's noise_level_param.
# For clarity with our execute_with_noise, we'll use a noise scaling method
# that directly modifies the circuit.
# The `noise_level_param` in `zne_base_exec_for_standalone_zne` will be effectively
# multiplied by the ZNE scale factor if we used a pass-through `scale_noise` function.
# However, with `fold_global`, the circuit itself is scaled.
# So, the `noise_level_param` in `zne_base_exec_for_standalone_zne` should be 1.0
# when used with circuit folding, as the folding itself is the scaling mechanism.

# Let's define a simpler executor for ZNE when using circuit folding,
# where noise_level_param is fixed at 1.0 because scaling is in the circuit.
executor_for_zne_folding = partial(execute_with_noise,
                                   noise_level_param=1.0, # Fixed, as circuit folding handles scaling
                                   rz_angle_param=base_rz_angle,
                                   idle_error_param=base_idle_error,
                                   p_readout_param=base_p_readout,
                                   depol_prob_param=base_depol_prob)


zne_executor_standalone = zne.mitigate_executor(
    executor_for_zne_folding, # This executor applies the base noise to already-scaled circuits
    observable=obs,
    scale_noise=zne.scaling.folding.fold_global, # Modifies circuit structure
    factory=zne_factory_standalone
)

zne_result_val = zne_executor_standalone(circuit).real # ZNE scales the 'circuit' then passes to executor
print(f"ZNE mitigated expectation value (Linear Fit): {zne_result_val:.6f}")
print(f"Absolute error after ZNE (Linear Fit): {abs(ideal_result_val - zne_result_val):.6f}")

# For visualizing different fits later
zne_data_points_x = zne_factory_standalone.get_scale_factors()
zne_data_points_y = zne_factory_standalone.get_expectation_values()
```

## Building the Advanced Error Mitigation Pipeline

Now, let's combine these techniques. The order of application can be crucial:
1.  **PT**: Applied first to the base circuit to tailor coherent noise.
2.  **DDD**: Applied to the Pauli-twirled circuits to mitigate time-correlated noise.
3.  **REM**: Wrapped around the executor that runs the PT+DDD modified circuits to correct readout errors.
4.  **ZNE**: Applied last, using the PT+DDD+REM setup as its "noisy executor", to extrapolate to the zero-noise limit.

### Approach 1: Sequential Executor Wrapping

This approach involves creating layers of executors, where each outer layer incorporates an additional mitigation technique.

```{code-cell} ipython3
# This executor will apply PT, DDD, and REM at a fixed noise level (1.0),
# and ZNE will scale the noise through circuit folding.
def full_mitigation_executor(circuit_to_run: cirq.Circuit, verbose=True) -> float:
    """Apply PT + DDD + REM techniques to a circuit and return the expectation value."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXECUTING PIPELINE ON CIRCUIT WITH {len(circuit_to_run.all_qubits())} QUBITS")
        print(f"Input circuit depth: {len(circuit_to_run)}")
        print(f"{'='*60}")
    
    # Define the base noisy executor using fixed noise parameters
    base_noisy_exec = partial(execute_with_noise,
                           noise_level_param=1.0,  # Fixed base noise level
                           rz_angle_param=base_rz_angle,
                           idle_error_param=base_idle_error,
                           p_readout_param=base_p_readout,
                           depol_prob_param=base_depol_prob)
    
    if verbose:
        print(f"\n[1/3] Applying REM with parameters: p0={base_p_readout}, p1={base_p_readout}")
    
    # 1. Apply REM to the base executor
    rem_exec = rem.mitigate_executor(
        base_noisy_exec, 
        inverse_confusion_matrix=inverse_confusion_matrix
    )
    
    if verbose:
        print(f"\n[2/3] Applying DDD with rule: {ddd_rule.__name__}")
    
    # 2. Apply DDD to the circuit
    ddd_circuit = ddd.insert_ddd_sequences(circuit_to_run, ddd_rule)
    
    if verbose:
        print(f"Circuit depth after DDD: {len(ddd_circuit)}")
    
    if verbose:
        print(f"\n[3/3] Applying PT with {num_twirled_variants} twirling variants")
    
    # 3. Apply PT to the DDD-modified circuit
    pt_variants = generate_pauli_twirl_variants(
        ddd_circuit, 
        num_circuits=num_twirled_variants,
        random_state=0  # Fixed seed for reproducibility
    )
    
    # Execute all PT+DDD variants with REM and average the results
    expectations = []
    
    if verbose:
        print(f"\nExecuting {num_twirled_variants} twirled circuits with REM-wrapped executor:")
        
    for i, c in enumerate(pt_variants):
        if verbose:
            print(f"  Executing PT variant {i+1}/{num_twirled_variants}...", end="")
        
        exp_val = obs.expectation(c, rem_exec).real
        expectations.append(exp_val)
        
        if verbose:
            print(f" exp_val = {exp_val:.6f}")
    
    result = np.mean(expectations)
    
    if verbose:
        print(f"\nFinal expectation value (averaged over {num_twirled_variants} PT variants): {result:.6f}")
        print(f"{'='*60}")
    
    return result

# Set up ZNE with circuit folding
pipeline_scale_factors = [1.0, 1.5, 2.0]
pipeline_zne_factory = LinearFactory(scale_factors=pipeline_scale_factors)

# Create the ZNE mitigated executor
full_pipeline_zne_executor_approach1 = zne.mitigate_executor(
    executor=full_mitigation_executor,  # This executor already applies PT+DDD+REM
    observable=None,  # We don't need to pass an observable as full_mitigation_executor returns a float
    scale_noise=zne.scaling.folding.fold_global,  # Scale noise by folding the circuit
    factory=pipeline_zne_factory
)

# Execute the pipeline
full_pipeline_result_val_approach1 = full_pipeline_zne_executor_approach1(circuit)
if hasattr(full_pipeline_result_val_approach1, 'real'):
    full_pipeline_result_val_approach1 = full_pipeline_result_val_approach1.real

print(f"Approach 1 Full pipeline (PT→DDD→REM→ZNE via circuit folding) result: {full_pipeline_result_val_approach1:.6f}")
print(f"Approach 1 Full pipeline absolute error: {abs(ideal_result_val - full_pipeline_result_val_approach1):.6f}")
```

### Approach 2: Structured Two-Stage Application (Conceptual for ZNE)

Mitiq's ZNE also supports a two-stage process: `zne.construct_circuits` and `factory.reduce`. We can structure our pipeline around this, though PT, DDD, and REM are applied differently.

*   **ZNE `construct_circuits`**: Generates circuits for different noise scale factors (e.g., by folding).
*   For each ZNE-scaled circuit:
    *   Apply DDD.
    *   Apply PT (generating multiple variants for each DDD+ZNE-scaled circuit).
    *   Execute these variants using a base executor (without REM).
    *   Apply REM to the raw measurement results.
    *   Average PT results (after REM) to get a single expectation value for that ZNE scale factor.
*   **ZNE `factory.reduce`**: Takes all averaged expectation values (one per ZNE scale factor) and performs the final extrapolation.

```{code-cell} ipython3
# Stage 1: ZNE Circuit Generation (using folding)
zne_scale_factors_approach2 = [1.0, 1.5, 2.0] 
scaled_circuits_by_zne_folding = zne.construct_circuits(
    circuit, 
    scale_factors=zne_scale_factors_approach2,
    scale_method=zne.scaling.folding.fold_global
)

all_scaled_expectations_for_zne_approach2 = []

# Base executor for circuit execution
base_noise_executor_for_approach2 = partial(execute_with_noise, 
                                          noise_level_param=1.0, 
                                          rz_angle_param=base_rz_angle,
                                          idle_error_param=base_idle_error,
                                          p_readout_param=base_p_readout,
                                          depol_prob_param=base_depol_prob)

# Process each ZNE-scaled circuit through DDD, PT, execute, then apply REM to results.
for zne_idx, zne_scaled_circuit in enumerate(scaled_circuits_by_zne_folding):
    # Apply DDD to the current ZNE-scaled circuit
    ddd_zne_scaled_circuit = ddd.insert_ddd_sequences(zne_scaled_circuit, ddd_rule)
    
    # Apply PT to the DDD-ZNE-scaled circuit
    pt_ddd_zne_scaled_circuits = generate_pauli_twirl_variants(
        ddd_zne_scaled_circuit, 
        num_circuits=num_twirled_variants,
        random_state=zne_idx
    )
    
    # Stage 2a: Execution and REM (per ZNE scale factor, averaged over PT variants)
    current_pt_expectations = []
    for pt_idx, final_circuit_variant in enumerate(pt_ddd_zne_scaled_circuits):
        # Execute to get raw measurement results
        raw_measurement_result = base_noise_executor_for_approach2(final_circuit_variant)
        
        # Apply REM to the raw measurement results
        from mitiq.rem.inverse_confusion_matrix import mitigate_measurements
        corrected_measurement_result = mitigate_measurements(
            raw_measurement_result,
            inverse_confusion_matrix
        )
        
        # Define a properly type-hinted executor function
        from typing import Any
        from mitiq import QPROGRAM, MeasurementResult
        
        def corrected_measurement_executor(_: QPROGRAM) -> MeasurementResult:
            return corrected_measurement_result
        
        # Calculate expectation using the type-hinted executor
        exp_val = obs.expectation(final_circuit_variant, corrected_measurement_executor).real
        current_pt_expectations.append(exp_val)
    
    # Average results for the current ZNE scale factor
    avg_exp_for_this_scale_factor = np.mean(current_pt_expectations)
    all_scaled_expectations_for_zne_approach2.append(avg_exp_for_this_scale_factor)

# Stage 2b: ZNE inference
zne_factory_approach2 = LinearFactory(scale_factors=zne_scale_factors_approach2)

# Push the data into the factory
for scale_factor, expectation in zip(zne_scale_factors_approach2, all_scaled_expectations_for_zne_approach2):
    zne_factory_approach2.push({"scale_factor": scale_factor}, expectation)

# Now call reduce() without arguments - it will use the data we just pushed
full_pipeline_result_val_approach2 = zne_factory_approach2.reduce()

print(f"Approach 2 Structured pipeline result: {full_pipeline_result_val_approach2:.6f}")
print(f"Approach 2 Structured pipeline absolute error: {abs(ideal_result_val - full_pipeline_result_val_approach2):.6f}")
```

## Visualizing ZNE Fits (Helper Function)

As suggested in the project goals, visualizing different ZNE fits can help choose the best extrapolation strategy. Let's create a helper function for this.

```{code-cell} ipython3
def visualize_zne_fits(scale_factors, results, ideal_value=None):
    """
    Plots ZNE data and overlays Linear, Richardson (if applicable),
    Polynomial (degree 2), and Exponential fits.
    """
    # Convert results to real values if they're complex
    results_real = [result.real if hasattr(result, 'real') else float(result) for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, results_real, "o-", label="Raw Data Points", markersize=8, color="black")

    # Linear Fit
    linear_factory = LinearFactory(scale_factors=scale_factors)
    # Push data into the factory
    for scale, result in zip(scale_factors, results):
        linear_factory.push({"scale_factor": scale}, result)
    linear_zne_val = linear_factory.reduce()  # No arguments needed here
    if hasattr(linear_zne_val, 'real'):
        linear_zne_val = linear_zne_val.real
    
    fit_x = np.linspace(0, max(scale_factors), 100)
    # Calculate fit manually using numpy's polyfit since we know LinearFactory uses a degree 1 polynomial
    fit_poly = np.polyfit(scale_factors, results_real, 1)
    fit_y_linear = np.polyval(fit_poly, fit_x)
    plt.plot(fit_x, fit_y_linear, "--", label=f"Linear Fit (ZNE: {linear_zne_val:.4f})")
    plt.plot(0, linear_zne_val, "X", markersize=10, label="Linear Extrapolation")

    # Similar approach for other factory types...
    # Richardson Extrapolation (if 2 or 3 points)
    if len(scale_factors) == 2 or len(scale_factors) == 3:
        try:
            richardson_factory = RichardsonFactory(scale_factors=scale_factors)
            # Push data into the factory
            for scale, result in zip(scale_factors, results):
                richardson_factory.push({"scale_factor": scale}, result)
            richardson_zne_val = richardson_factory.reduce()
            if hasattr(richardson_zne_val, 'real'):
                richardson_zne_val = richardson_zne_val.real
            
            # Richardson is equivalent to a polynomial fit of degree = len(scale_factors) - 1
            fit_poly = np.polyfit(scale_factors, results_real, len(scale_factors) - 1)
            fit_y_richardson = np.polyval(fit_poly, fit_x)
            plt.plot(fit_x, fit_y_richardson, ":", label=f"Richardson Fit (ZNE: {richardson_zne_val:.4f})")
            plt.plot(0, richardson_zne_val, "P", markersize=10, label="Richardson Extrapolation")
        except Exception as e:
            print(f"Could not plot Richardson: {e}")

    # Polynomial Fit (degree 2, if at least 3 points)
    if len(scale_factors) >= 3:
        poly_factory = PolyFactory(scale_factors=scale_factors, order=2)
        # Push data into the factory
        for scale, result in zip(scale_factors, results):
            poly_factory.push({"scale_factor": scale}, result)
        poly_zne_val = poly_factory.reduce()
        if hasattr(poly_zne_val, 'real'):
            poly_zne_val = poly_zne_val.real
        
        # Direct polynomial fit of degree 2
        fit_poly = np.polyfit(scale_factors, results_real, 2)
        fit_y_poly = np.polyval(fit_poly, fit_x)
        plt.plot(fit_x, fit_y_poly, "-.", label=f"Poly Fit (deg=2, ZNE: {poly_zne_val:.4f})")
        plt.plot(0, poly_zne_val, "D", markersize=10, label="Poly Extrapolation")

    # Exponential Fit (if at least 2 points)
    if len(scale_factors) >= 2:
        try:
            # Attempt to guess a reasonable asymptote if not 0 or 1
            asymptote_guess = 0.0 if ideal_value is None else (ideal_value / 2.0 if ideal_value > 0.2 else 0.0)
            if abs(min(results_real)) > abs(max(results_real)):  # if results tend to negative
                asymptote_guess = np.mean(results_real) / 2.0

            exp_factory = ExpFactory(scale_factors=scale_factors, asymptote=asymptote_guess)
            # Push data into the factory
            for scale, result in zip(scale_factors, results):
                exp_factory.push({"scale_factor": scale}, result)
            exp_zne_val = exp_factory.reduce()
            if hasattr(exp_zne_val, 'real'):
                exp_zne_val = exp_zne_val.real
            
            # For exponential fit, we skip manual curve generation and just show the point
            plt.plot(0, exp_zne_val, "s", markersize=10, label=f"Exp Extrapolation (ZNE: {exp_zne_val:.4f})")
        except Exception as e:
            print(f"Could not plot Exponential: {e}")

    if ideal_value is not None:
        if hasattr(ideal_value, 'real'):
            ideal_value = ideal_value.real
        plt.axhline(ideal_value, color="green", linestyle="--", label=f"Ideal Value ({ideal_value:.4f})")

    plt.xlabel("Noise Scale Factor")
    plt.ylabel("Expectation Value")
    plt.title("ZNE Data and Various Extrapolation Fits")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", alpha=0.7)
    
    # Use real values for y-axis limits
    plt.ylim(min(results_real) - 0.2 * abs(min(results_real)), max(results_real) + 0.2 * abs(max(results_real)))
    plt.xlim(-0.1, max(scale_factors) + 0.1)
    plt.gca().invert_xaxis()  # Common to plot 0 on the right for ZNE
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer ticks if possible
    plt.show()

# Visualize fits for the standalone ZNE data
print("\nVisualizing fits for Standalone ZNE data:")
visualize_zne_fits(zne_data_points_x, zne_data_points_y, ideal_value=ideal_result_val)

# Visualize fits for Approach 1 Pipeline ZNE data
print("\nVisualizing fits for Approach 1 Pipeline ZNE data:")
pipeline_zne_data_x_approach1 = pipeline_zne_factory.get_scale_factors()
pipeline_zne_data_y_approach1 = pipeline_zne_factory.get_expectation_values()
visualize_zne_fits(pipeline_zne_data_x_approach1, pipeline_zne_data_y_approach1, ideal_value=ideal_result_val)

# Visualize fits for Approach 2 Pipeline ZNE data
print("\nVisualizing fits for Approach 2 Pipeline ZNE data:")
# We already have: zne_scale_factors_approach2 and all_scaled_expectations_for_zne_approach2
visualize_zne_fits(zne_scale_factors_approach2, all_scaled_expectations_for_zne_approach2, ideal_value=ideal_result_val)

```

## Comparing Results

Let's summarize the expectation values and errors obtained.

```{code-cell} ipython3
results_summary = {
    "Ideal": ideal_result_val,
    "Unmitigated Noisy": noisy_result_val,
    "PT only": pt_result_val,
    "DDD only": ddd_result_val,
    "REM only": rem_result_val,
    "ZNE only (Linear)": zne_result_val,
    "Pipeline (Approach 1 - Linear ZNE)": full_pipeline_result_val_approach1,
    "Pipeline (Approach 2 - Linear ZNE)": full_pipeline_result_val_approach2,
}

print("Summary of Expectation Values and Errors:")
print("-------------------------------------------")
for name, val in results_summary.items():
    error = abs(ideal_result_val - val)
    print(f"{name:<35}: Value = {val:+.6f}, Abs Error = {error:.6f}")
```

## Visualizing Overall Improvements

A bar chart can effectively illustrate the reduction in error at each stage and with the full pipeline.

```{code-cell} ipython3
labels = list(results_summary.keys())
values = [results_summary[label] for label in labels]
errors_viz = [abs(ideal_result_val - val) for val in values]

x_pos = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(16, 8)) # Increased figure size

# Bar chart for absolute error
color_error = 'salmon'
ax1.set_xlabel('Mitigation Strategy', fontsize=12)
ax1.set_ylabel('Absolute Error (from Ideal)', color=color_error, fontsize=12)
bars_error = ax1.bar(x_pos, errors_viz, width=0.6, label='Absolute Error', color=color_error, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color_error, labelsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

# Line plot for expectation values on a secondary y-axis
ax2 = ax1.twinx()
color_value = 'steelblue'
ax2.set_ylabel('Expectation Value', color=color_value, fontsize=12)
ax2.plot(x_pos, values, 'o-', label='Expectation Value', color=color_value, markersize=7, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_value, labelsize=10)
ax2.axhline(ideal_result_val, color='darkgreen', linestyle='--', linewidth=2, label=f'Ideal Value ({ideal_result_val:.3f})')

# Add data labels on bars
for bar in bars_error:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=8, color='black')


fig.tight_layout() # Adjust layout to prevent overlapping labels
plt.title('Comparison of Error Mitigation Strategies and Their Impact', fontsize=14)

# Combine legends
lines, labels_ax1_leg = ax1.get_legend_handles_labels()
lines2, labels_ax2_leg = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels_ax1_leg + labels_ax2_leg, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10)


plt.show()
```

## Conclusion

This tutorial demonstrated how to construct an advanced error mitigation pipeline by combining Pauli Twirling (PT), Digital Dynamical Decoupling (DDD), Readout Error Mitigation (REM), and Zero-Noise Extrapolation (ZNE).

Key takeaways:
*   **Individual Impact**: Each technique showed some improvement over the unmitigated noisy result, targeting different error sources. The degree of improvement depends heavily on the nature and magnitude of the corresponding noise types in the system.
*   **Pipeline Power**: The full pipeline, leveraging the strengths of all four techniques in a considered order, generally yields the best result, bringing the expectation value significantly closer to the ideal. The exact improvement will vary based on the noise model and circuit.
*   **Ordering and Interaction**: The order of application (e.g., PT before ZNE, DDD on circuits before execution, REM on results or as an executor wrapper) is designed to be effective. Techniques can interact; for instance, PT can make noise more amenable to ZNE.
*   **Flexibility in Implementation**: Mitiq's tools can be composed in different ways. We explored:
    *   **Approach 1 (Sequential Executor Wrapping for ZNE)**: ZNE scales noise by passing a `noise_level_param` to a comprehensive executor that internally handles PT, DDD, and REM for each ZNE evaluation point.
    *   **Approach 2 (Structured Two-Stage for ZNE)**: ZNE scales noise by circuit folding (`construct_circuits`). Then, for each ZNE-scaled circuit, DDD and PT are applied, followed by execution with an REM-enabled executor. Finally, ZNE's `reduce` function performs the extrapolation.
*   **Complexity and Cost**: Implementing such a pipeline involves careful management of circuit transformations, executor compositions, and potentially a significant increase in the number of circuit executions (especially due to PT and ZNE).
*   **ZNE Fit Visualization**: The `visualize_zne_fits` helper function demonstrates how one can inspect the ZNE data and compare different extrapolation models, aiding in the selection of an appropriate `Factory`.

The specific improvements and optimal configuration of such a pipeline can depend heavily on the dominant noise types in a given quantum device and the chosen circuit. Experimentation and characterization are key to tailoring effective error mitigation strategies. This tutorial provides a template for building and evaluating such combined approaches with Mitiq.

```{code-cell} ipython3
# Display Mitiq version information
mitiq.about()
```