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
4.  **Zero-Noise Extrapolation (ZNE)**: Suppresses generic gate noise by extrapolating results from circuits run at amplified noise levels back to the zero-noise limit.

We'll implement a step-by-step approach, analyzing the impact of each technique individually. For DDD, REM, and ZNE, we will highlight the pattern of constructing circuits (or preparing models) and then combining results after execution. Finally, we'll demonstrate how to effectively combine all techniques into a single pipeline.

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
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
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
    rz_angle_param: float = 0.01,    # Magnitude of coherent RZ error
    idle_error_param: float = 0.005, # Magnitude of time-correlated Z error during idle periods
    p_readout_param: float = 0.008,  # Probability of readout bit-flip
    depol_prob_param: float = 0.004, # Probability of depolarizing error
    repetitions: int = 4000 # Number of shots
) -> MeasurementResult:
    """
    Executes a circuit with a comprehensive noise model.
    The noise_level_param is primarily for ZNE to scale the overall noise impact.
    """
    noisy_circuit = circuit_to_run.copy()
    qubits = sorted(noisy_circuit.all_qubits())

    current_rz_angle = rz_angle_param * noise_level_param
    current_idle_error = idle_error_param * noise_level_param
    current_p_readout = p_readout_param * noise_level_param
    current_depol_prob = depol_prob_param * noise_level_param

    current_p_readout = min(current_p_readout, 1.0)
    current_depol_prob = min(current_depol_prob, 1.0)

    temp_circuit_ops = []

    for moment_idx, moment in enumerate(noisy_circuit.moments):
        temp_circuit_ops.append(moment)
        if current_rz_angle > 0:
            temp_circuit_ops.append(cirq.Moment(cirq.rz(rads=current_rz_angle).on(q) for q in qubits))
        if current_idle_error > 0:
             temp_circuit_ops.append(cirq.Moment(cirq.Z(q)**( (moment_idx + 1) * current_idle_error / len(noisy_circuit.moments) ) for q in qubits))

    noisy_circuit_with_moment_noise = cirq.Circuit(temp_circuit_ops)

    if current_depol_prob > 0:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise.with_noise(cirq.depolarize(p=current_depol_prob))
    else:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise

    if current_p_readout > 0:
        noisy_circuit_with_depol.append(cirq.bit_flip(p=current_p_readout).on_each(*qubits))

    noisy_circuit_with_depol.append(cirq.measure(*qubits, key='m'))

    simulator = cirq.DensityMatrixSimulator() # Using DensityMatrixSimulator for noise modeling
    result = simulator.run(noisy_circuit_with_depol, repetitions=repetitions)

    bitstrings = result.measurements['m']

    return MeasurementResult(bitstrings)
```

## Establishing Baselines

First, let's determine the ideal (noiseless) expectation value and the unmitigated noisy expectation value with our adjusted (lower) noise settings.

```{code-cell} ipython3
noiseless_exec = partial(execute_with_noise,
                         noise_level_param=0.0, # Turn off all ZNE-scalable noise components
                         rz_angle_param=0.0,    # Turn off coherent phase error
                         idle_error_param=0.0,  # Turn off time-correlated noise
                         p_readout_param=0.0,   # Turn off readout error
                         depol_prob_param=0.0)  # Turn off depolarizing noise


ideal_result_val = obs.expectation(circuit, noiseless_exec).real
print(f"Ideal expectation value: {ideal_result_val:.6f}")

# Base noise strengths for mitigation experiments
base_rz_angle = 0.00585    
base_idle_error = 0.00297 
base_p_readout = 0.00468   
base_depol_prob = 0.00234

noisy_exec = partial(execute_with_noise,
                     noise_level_param=1.0, 
                     rz_angle_param=base_rz_angle,
                     idle_error_param=base_idle_error,
                     p_readout_param=base_p_readout,
                     depol_prob_param=base_depol_prob)

noisy_result_val = obs.expectation(circuit, noisy_exec).real
print(f"Unmitigated noisy expectation value: {noisy_result_val:.6f}")
print(f"Initial absolute error: {abs(ideal_result_val - noisy_result_val):.6f}")
```

## Applying Individual Error Mitigation Techniques

Now, let's apply each technique individually to observe its impact. The `noisy_exec` defined above (with `noise_level_param=1.0` and reduced base noise parameters) will be used as the baseline noisy executor for these individual tests.

### 1. Pauli Twirling (PT)

Pauli Twirling aims to convert coherent noise into stochastic Pauli noise.

```{code-cell} ipython3
num_twirled_variants = 10 # Number of twirled circuits to average over (can be adjusted)
twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=num_twirled_variants, random_state=0)

pt_expectations = []
for tw_circuit_idx, tw_circuit in enumerate(twirled_circuits):
    print(f"Executing PT variant {tw_circuit_idx+1}/{num_twirled_variants}") # Optional: for progress tracking
    exp_val = obs.expectation(tw_circuit, noisy_exec).real
    pt_expectations.append(exp_val)

pt_result_val = np.mean(pt_expectations)
print(f"PT mitigated expectation value: {pt_result_val:.6f}")
print(f"Absolute error after PT: {abs(ideal_result_val - pt_result_val):.6f}")
```

### 2. Digital Dynamical Decoupling (DDD)

DDD inserts sequences of pulses to decouple qubits from certain types of environmental noise.

```{code-cell} ipython3
ddd_rule = rules.xyxy
ddd_circuit = ddd.insert_ddd_sequences(circuit, ddd_rule)
ddd_result_val = obs.expectation(ddd_circuit, noisy_exec).real
print(f"DDD mitigated expectation value: {ddd_result_val:.6f}")
print(f"Absolute error after DDD: {abs(ideal_result_val - ddd_result_val):.6f}")
```

### 3. Readout Error Mitigation (REM)

REM corrects errors that occur during the measurement process.

```{code-cell} ipython3
p0_rem = base_p_readout # P(1|0)
p1_rem = base_p_readout # P(0|1)

# The confusion matrix is our "model"
inverse_confusion_matrix = rem.generate_inverse_confusion_matrix(
    num_qubits, p0=p0_rem, p1=p1_rem
)

raw_measurement_result_for_rem = noisy_exec(circuit)

mitigated_measurement_result = rem.mitigate_measurements(
    raw_measurement_result_for_rem, 
    inverse_confusion_matrix
)

rem_result_val = obs._expectation_from_measurements([mitigated_measurement_result]).real

print(f"REM mitigated expectation value: {rem_result_val:.6f}")
print(f"Absolute error after REM: {abs(ideal_result_val - rem_result_val):.6f}")
```

### 4. Zero-Noise Extrapolation (ZNE)

ZNE runs the circuit at different amplified noise levels and extrapolates the results back to the zero-noise limit.

```{code-cell} ipython3
scale_factors_zne = [1.0, 1.5, 2.0]

scaled_circuits_zne = zne.construct_circuits(
    circuit, 
    scale_factors=scale_factors_zne,
    scale_method=fold_global
)

scaled_expectations_zne = []
for sc in scaled_circuits_zne:
    result = noisy_exec(sc)
    # Calculate expectation value from measurement result
    exp_val = obs._expectation_from_measurements([result]).real
    scaled_expectations_zne.append(exp_val)

def linear_extrapolation(scale_factors, expectation_values):
    factory = LinearFactory(scale_factors=scale_factors)
    for sf, val in zip(scale_factors, expectation_values):
        factory.push({"scale_factor": sf}, val)
    return factory.reduce()

zne_result_val = zne.combine_results(
    scale_factors_zne,
    scaled_expectations_zne,
    extrapolation_method=linear_extrapolation
)
if hasattr(zne_result_val, 'real'):
    zne_result_val = zne_result_val.real

print(f"ZNE mitigated expectation value (Linear Fit): {zne_result_val:.6f}")
print(f"Absolute error after ZNE (Linear Fit): {abs(ideal_result_val - zne_result_val):.6f}")
```

## Building the Full Error Mitigation Pipeline

Now, let's combine these techniques into a single, comprehensive pipeline. The order of application will be:
1. **ZNE `construct_circuits`**: Create noise-scaled versions of the original circuit.
2. **DDD `construct_circuits`**: Apply DDD sequences to each ZNE-scaled circuit.
3. **PT**: Generate Pauli twirled variants for each DDD-modified, ZNE-scaled circuit.
4. Execute all these variants.
5. **REM `combine_results`**: Apply readout correction to the execution results.
6. **PT, DDD averaging**: Combine the REM-corrected results for PT variants and then DDD variants.
7. **ZNE `combine_results`**: Extrapolate to zero noise using the results from different scale factors.

```{code-cell} ipython3
pipeline_scale_factors = [1.0, 1.5, 2.0]

print(f"\nEXECUTING FULL PIPELINE (ZNE→DDD→PT→REM)")
print(f"{'='*60}")

zne_scaled_circuits = zne.construct_circuits(
    circuit,
    scale_factors=pipeline_scale_factors,
    scale_method=fold_global
)
print(f"ZNE: Generated {len(zne_scaled_circuits)} scaled circuits with factors {pipeline_scale_factors}")

all_results = []

for sf_idx, zne_circuit in enumerate(zne_scaled_circuits):
    scale_factor = pipeline_scale_factors[sf_idx]
    print(f"\nProcessing ZNE scale factor: {scale_factor}")

    ddd_circuits = ddd.construct_circuits(zne_circuit, rule=ddd_rule)
    print(f"  DDD: Generated {len(ddd_circuits)} circuits")

    ddd_results = []

    for ddd_idx, ddd_circuit in enumerate(ddd_circuits):
        pt_circuits = generate_pauli_twirl_variants(
            ddd_circuit,
            num_circuits=num_twirled_variants,
            random_state=sf_idx+ddd_idx 
        )
        print(f"  PT: Generated {len(pt_circuits)} variants for DDD circuit {ddd_idx+1}")

        pt_results = []
        
        for pt_circuit in pt_circuits:
            raw_result = noisy_exec(pt_circuit)
            mitigated_result = rem.mitigate_measurements(  
                raw_result, 
                inverse_confusion_matrix
            )
            pt_results.append(mitigated_result)
        
        exp_val_for_one_ddd_circuit = obs._expectation_from_measurements(pt_results).real  
        ddd_results.append(exp_val_for_one_ddd_circuit)
    
    exp_val_for_sf = ddd.combine_results(ddd_results) 
    all_results.append(exp_val_for_sf)
    
    
    print(f"  Scale factor {scale_factor} expectation: {exp_val_for_sf:.6f}")

def linear_extrapolation(scale_factors, expectation_values):
    factory = LinearFactory(scale_factors=scale_factors)
    for sf, val in zip(scale_factors, expectation_values):
        factory.push({"scale_factor": sf}, val)
    return factory.reduce()

full_pipeline_result_val = zne.combine_results(
    pipeline_scale_factors,
    all_results,
    extrapolation_method=linear_extrapolation
)
if hasattr(full_pipeline_result_val, 'real'):
    full_pipeline_result_val = full_pipeline_result_val.real

print(f"{'='*60}")
print(f"\nFull pipeline (ZNE→DDD→PT→REM) result: {full_pipeline_result_val:.6f}")
print(f"Full pipeline absolute error: {abs(ideal_result_val - full_pipeline_result_val):.6f}")
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
    "Full Pipeline": full_pipeline_result_val,
}

print("\nSummary of Expectation Values and Errors:")
print("-------------------------------------------")
for name, val_obj in results_summary.items():
    val = val_obj.real if hasattr(val_obj, 'real') else float(val_obj)
    error = abs(ideal_result_val - val) 
    print(f"{name:<35}: Value = {val:+.6f}, Abs Error = {error:.6f}")
```

## Visualizing Overall Improvements

A bar chart can effectively illustrate the reduction in error at each stage and with the full pipeline.

```{code-cell} ipython3
labels = list(results_summary.keys())
values_for_plot = [v.real if hasattr(v, 'real') else float(v) for v in results_summary.values()]
errors_viz = [abs(ideal_result_val - val) for val in values_for_plot] # ideal_result_val is already real

x_pos = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(14, 7)) 

color_error = 'salmon'
ax1.set_xlabel('Mitigation Strategy', fontsize=12)
ax1.set_ylabel('Absolute Error (from Ideal)', color=color_error, fontsize=12)
bars_error = ax1.bar(x_pos, errors_viz, width=0.6, label='Absolute Error', color=color_error, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color_error, labelsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

ax2 = ax1.twinx()
color_value = 'steelblue'
ax2.set_ylabel('Expectation Value', color=color_value, fontsize=12)
ax2.plot(x_pos, values_for_plot, 'o-', label='Expectation Value', color=color_value, markersize=7, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_value, labelsize=10)
ax2.axhline(ideal_result_val, color='darkgreen', linestyle='--', linewidth=2, label=f'Ideal Value ({ideal_result_val:.3f})') # ideal_result_val is real

for bar_idx, bar in enumerate(bars_error):
    yval = bar.get_height()
    # Check if the corresponding expectation value is negative to adjust label position slightly for clarity
    is_negative_exp_val = values_for_plot[bar_idx] < 0
    offset = 0.01 if is_negative_exp_val and yval < 0.1 else 0.005 
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.4f}', ha='center', va='bottom', fontsize=8, color='black')

fig.tight_layout()
plt.title('Comparison of Error Mitigation Strategies and Their Impact', fontsize=14)
lines_ax1, labels_ax1_leg = ax1.get_legend_handles_labels()
lines_ax2, labels_ax2_leg = ax2.get_legend_handles_labels()
ax2.legend(lines_ax1 + lines_ax2, labels_ax1_leg + labels_ax2_leg, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=10)

plt.show()
```

## Conclusion

This tutorial demonstrated how to construct an advanced error mitigation pipeline by combining Pauli Twirling (PT), Digital Dynamical Decoupling (DDD), Readout Error Mitigation (REM), and Zero-Noise Extrapolation (ZNE).

Key takeaways:
*   **Run-to-Run Variability**: Due to the stochastic nature of quantum noise and finite measurement shots, results can vary between runs. Users might observe different relative performance of techniques when executing this notebook.

*   **Range of Effectiveness**: In our experiments, we typically observed that ZNE alone and the Full Pipeline provided the most significant error reduction, while PT alone sometimes performed worse than the unmitigated case.

*   **API Pattern Consistency**: For DDD, REM, and ZNE, we followed a consistent pattern of first constructing necessary circuits or models (`construct_circuits`), then executing them, and finally combining the results (`combine_results`).

*   **Implementation Complexity vs. Benefit**: The full pipeline requires significantly more circuit executions. Users should evaluate whether this computational overhead is justified by any potential accuracy improvements for their specific applications.

This tutorial provides a framework for experimenting with combined error mitigation approaches in Mitiq. For optimal results in real applications, it's recommended to first characterize the noise on your quantum hardware and then strategically select and combine the most effective techniques for your specific noise profile.

```{code-cell} ipython3
# Display Mitiq version information
mitiq.about()
```