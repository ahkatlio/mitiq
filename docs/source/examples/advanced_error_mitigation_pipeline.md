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

```{tags} pt, ddd, rem, zne, advanced, cirq
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
from mitiq import pt, ddd, rem, zne 
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
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
# For an ideal GHZ state |Ψ⟩ = (|0000⟩ + |1111⟩)/√2, 
# the expectation value ⟨Ψ|X⊗X⊗X⊗X|Ψ⟩ is 1.
obs = Observable(PauliString("X" * num_qubits))
print(f"\nObservable: {obs}")
```

## Comprehensive Noise Model

To demonstrate the benefits of each mitigation technique, we need a noise model that incorporates various error sources. Our model uses parameters that are representative of noise levels seen in current superconducting quantum processors:

- **Coherent phase errors**: ~0.01 radians (~0.57 degrees) corresponds to realistic over/under-rotation errors in single-qubit gates on many hardware platforms.
- **Time-correlated idle noise**: ~0.005 amplitude represents T2-like dephasing that accumulates during circuit execution, similar to what's observed on superconducting systems.
- **Readout errors**: ~0.008 bit-flip probability per qubit is comparable to readout fidelities of 99.2%, which is achievable on high-quality qubits.
- **Depolarizing noise**: ~0.004 probability is in line with single-qubit gate error rates on state-of-the-art hardware.

These values are deliberately chosen to be somewhat optimistic but realistic, representing a high-quality near-term device where error mitigation techniques would provide meaningful benefits without completely overwhelming the quantum signal.

```{code-cell} ipython3
def execute_with_noise(
    circuit_to_run: cirq.Circuit,
    noise_level_param: float = 1.0,  # General scaling for ZNE
    rz_angle_param: float = 0.01,    # Coherent over-rotation ~0.01 radians ≈ 0.57°
    idle_error_param: float = 0.005, # 0.5% phase error per idle step
    p_readout_param: float = 0.008,  # 0.8% readout bit-flip error
    depol_prob_param: float = 0.004, # 0.4% depolarizing probability
    repetitions: int = 4000          # Number of shots
) -> MeasurementResult:
    """
    Executes a circuit with a comprehensive noise model.
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
            temp_circuit_ops.append(
                cirq.Moment(
                    cirq.rz(rads=current_rz_angle).on(q) for q in qubits
                )
            )
        if current_idle_error > 0:
            error_factor = (moment_idx + 1) * current_idle_error / len(noisy_circuit.moments)
            temp_circuit_ops.append(
                cirq.Moment(
                    cirq.Z(q)**(error_factor) for q in qubits
                )
            )

    noisy_circuit_with_moment_noise = cirq.Circuit(temp_circuit_ops)

    if current_depol_prob > 0:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise.with_noise(
            cirq.depolarize(p=current_depol_prob)
        )
    else:
        noisy_circuit_with_depol = noisy_circuit_with_moment_noise

    if current_p_readout > 0:
        noisy_circuit_with_depol.append(
            cirq.bit_flip(p=current_p_readout).on_each(*qubits)
        )

    noisy_circuit_with_depol.append(cirq.measure(*qubits, key='m'))

    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(noisy_circuit_with_depol, repetitions=repetitions)

    bitstrings = result.measurements['m']

    return MeasurementResult(bitstrings)
```

## Establishing Baselines

First, let's determine the ideal (noiseless) expectation value and the unmitigated noisy expectation value with our adjusted (lower) noise settings.

```{code-cell} ipython3
noiseless_exec = partial(
    execute_with_noise,
    noise_level_param=0.0,  # Turn off all ZNE-scalable noise components
    rz_angle_param=0.0,     # Turn off coherent phase error
    idle_error_param=0.0,   # Turn off time-correlated noise
    p_readout_param=0.0,    # Turn off readout error
    depol_prob_param=0.0    # Turn off depolarizing noise
)

ideal_result_val = obs.expectation(circuit, noiseless_exec).real
print(f"Ideal expectation value: {ideal_result_val:.6f}")

noisy_exec = partial(execute_with_noise)

noisy_result_val = obs.expectation(circuit, noisy_exec).real
print(f"Unmitigated noisy expectation value: {noisy_result_val:.6f}")
print(f"Initial absolute error: {abs(ideal_result_val - noisy_result_val):.6f}")
```

## Applying Individual Error Mitigation Techniques

Now, let's apply each technique individually to observe its impact. The `noisy_exec` defined above (with `noise_level_param=1.0` and reduced base noise parameters) will be used as the baseline noisy executor for these individual tests.

### 1. Pauli Twirling (PT)

Pauli Twirling aims to convert coherent noise into stochastic Pauli noise.

```{code-cell} ipython3
# Number of twirled circuits to average over (can be adjusted)
num_twirled_variants = 10
twirled_circuits = pt.generate_pauli_twirl_variants(
    circuit, 
    num_circuits=num_twirled_variants, 
    random_state=0
)

pt_expectations = []
for tw_circuit_idx, tw_circuit in enumerate(twirled_circuits):
    print(f"Executing PT variant {tw_circuit_idx+1}/{num_twirled_variants}")
    exp_val = obs.expectation(tw_circuit, noisy_exec).real
    pt_expectations.append(exp_val)

pt_result_val = np.mean(pt_expectations)
print(f"PT mitigated expectation value: {pt_result_val:.6f}")
print(f"Absolute error after PT: {abs(ideal_result_val - pt_result_val):.6f}")
```

### 2. Digital Dynamical Decoupling (DDD)

DDD inserts sequences of pulses to decouple qubits from certain types of environmental noise.

```{code-cell} ipython3
ddd_circuit = ddd.insert_ddd_sequences(circuit, ddd.rules.xyxy)
ddd_result_val = obs.expectation(ddd_circuit, noisy_exec).real
print(f"DDD mitigated expectation value: {ddd_result_val:.6f}")
print(f"Absolute error after DDD: {abs(ideal_result_val - ddd_result_val):.6f}")
```

### 3. Readout Error Mitigation (REM)

REM corrects errors that occur during the measurement process.

```{code-cell} ipython3
p0_rem = 0.008  # P(1|0)
p1_rem = 0.008  # P(0|1)

# The confusion matrix is our "model"
inverse_confusion_matrix = rem.generate_inverse_confusion_matrix(
    num_qubits, p0=p0_rem, p1=p1_rem
)

raw_measurement_result_for_rem = noisy_exec(circuit)

mitigated_measurement_result = rem.mitigate_measurements(
    raw_measurement_result_for_rem, 
    inverse_confusion_matrix
)

rem_result_val = obs._expectation_from_measurements(
    [mitigated_measurement_result]
).real

print(f"REM mitigated expectation value: {rem_result_val:.6f}")
print(f"Absolute error after REM: {abs(ideal_result_val - rem_result_val):.6f}")
```

### 4. Zero-Noise Extrapolation (ZNE)

ZNE runs the circuit at different amplified noise levels and extrapolates the results back to the zero-noise limit.

```{code-cell} ipython3
scale_factors = [1, 1.5, 2]

scaled_circuits_zne = zne.construct_circuits(
    circuit, 
    scale_factors=scale_factors,
    scale_method=fold_global
)

scaled_expectations_zne = []
for sc in scaled_circuits_zne:
    result = noisy_exec(sc)
    exp_val = obs._expectation_from_measurements([result]).real
    scaled_expectations_zne.append(exp_val)

zne_result_val = zne.combine_results(
    scale_factors,
    scaled_expectations_zne,
    extrapolation_method=LinearFactory.extrapolate
)
if hasattr(zne_result_val, 'real'):
    zne_result_val = zne_result_val.real

print(f"ZNE mitigated expectation value (Linear Fit): {zne_result_val:.6f}")
print(f"Absolute error after ZNE (Linear Fit): {abs(ideal_result_val - zne_result_val):.6f}")
```

## Combining REM and ZNE

Given that REM and ZNE often provide significant improvements, let's test their combined effect. The pipeline will be ZNE -> REM.

```{code-cell} ipython3
print(f"\nEXECUTING REM+ZNE COMBINATION (ZNE→REM)")
print(f"{'='*60}")

rem_zne_scaled_circuits = zne.construct_circuits(
    circuit, 
    scale_factors=scale_factors,
    scale_method=fold_global
)
print(f"ZNE: Generated {len(rem_zne_scaled_circuits)} scaled circuits with factors {scale_factors}")

rem_zne_scaled_expectations = []
for scale_factor, zne_scaled_circuit in zip(scale_factors, rem_zne_scaled_circuits):
    print(f"  Processing ZNE scale factor: {scale_factor}")
    
    raw_measurement_result = noisy_exec(zne_scaled_circuit)
    
    rem_corrected_measurement_result = rem.mitigate_measurements(
        raw_measurement_result, 
        inverse_confusion_matrix
    )
    
    exp_val_after_rem = obs._expectation_from_measurements(
        [rem_corrected_measurement_result]
    ).real
    
    rem_zne_scaled_expectations.append(exp_val_after_rem)
    print(f"  Scale factor {scale_factor} expectation (after REM): {exp_val_after_rem:.6f}")

rem_zne_pipeline_result_val = zne.combine_results(
    scale_factors,
    rem_zne_scaled_expectations,
    extrapolation_method=LinearFactory.extrapolate
)
if hasattr(rem_zne_pipeline_result_val, 'real'):
    rem_zne_pipeline_result_val = rem_zne_pipeline_result_val.real

print(f"{'='*60}")
print(f"\nREM+ZNE pipeline result: {rem_zne_pipeline_result_val:.6f}")
print(
    f"REM+ZNE pipeline absolute error: "
    f"{abs(ideal_result_val - rem_zne_pipeline_result_val):.6f}"
)
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
print(f"\nEXECUTING FULL PIPELINE (ZNE→DDD→PT→REM)")
print(f"{'='*60}")

zne_scaled_circuits = zne.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    scale_method=fold_global
)
print(
    f"ZNE: Generated {len(zne_scaled_circuits)} scaled circuits "
    f"with factors {scale_factors}"
)

all_results = []
for scale_factor, scaled_circuit in zip(scale_factors, zne_scaled_circuits):
    print(f"\nProcessing ZNE scale factor: {scale_factor}")

    ddd_circuits = ddd.construct_circuits(scaled_circuit, rule=ddd.rules.xyxy)
    print(f"  DDD: Generated {len(ddd_circuits)} circuits")

    ddd_results = []

    for ddd_idx, ddd_circuit in enumerate(ddd_circuits):
        pt_circuits = pt.generate_pauli_twirl_variants(
            ddd_circuit,
            num_circuits=num_twirled_variants,
            random_state=ddd_idx
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

full_pipeline_result_val = zne.combine_results(
    scale_factors,
    all_results,
    extrapolation_method=LinearFactory.extrapolate 
)
if hasattr(full_pipeline_result_val, 'real'):
    full_pipeline_result_val = full_pipeline_result_val.real

print(f"{'='*60}")
print(f"\nFull pipeline (ZNE→DDD→PT→REM) result: {full_pipeline_result_val:.6f}")
print(f"Full pipeline absolute error: {abs(ideal_result_val - full_pipeline_result_val):.6f}")
```

### Building the Full Error Mitigation Pipeline (Swapped Order: ZNE→PT→DDD→REM)

Now, let's try swapping the order of Pauli Twirling (PT) and Digital Dynamical Decoupling (DDD).
The new order of application will be:
1.  **ZNE `construct_circuits`**: Create noise-scaled versions of the original circuit.
2.  **PT `generate_pauli_twirl_variants`**: Generate Pauli twirled variants for each ZNE-scaled circuit.
3.  **DDD `construct_circuits`**: Apply DDD sequences to each PT-modified, ZNE-scaled circuit.
4.  Execute all these variants.
5.  **REM `mitigate_measurements`**: Apply readout correction to the execution results of each variant.
6.  **DDD averaging, then PT averaging**: For each PT variant, average the REM-corrected results from its DDD sub-variants. Then, average the results across all PT variants.
7.  **ZNE `combine_results`**: Extrapolate to zero noise using the results from different scale factors.

```{code-cell} ipython3
print(f"\nEXECUTING FULL PIPELINE WITH SWAPPED ORDER (ZNE→PT→DDD→REM)")
print(f"{'='*70}")

zne_scaled_circuits_swapped = zne.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    scale_method=fold_global
)
print(
    f"ZNE: Generated {len(zne_scaled_circuits_swapped)} scaled circuits "
    f"with factors {scale_factors}"
)

all_results_zne_level_swapped = [] 

for scale_factor_swapped, scaled_circuit_swapped in zip(scale_factors, zne_scaled_circuits_swapped):
    print(
        f"\nProcessing ZNE scale factor: {scale_factor_swapped} "
        f"(Swapped Order ZNE→PT→DDD→REM)"
    )

    pt_variants_of_zne_circuit = pt.generate_pauli_twirl_variants(
        scaled_circuit_swapped,
        num_circuits=num_twirled_variants, 
        random_state=scale_factors.index(scale_factor_swapped) 
    )
    print(
        f"  PT: Generated {len(pt_variants_of_zne_circuit)} variants "
        f"for ZNE scale factor {scale_factor_swapped}"
    )

    pt_level_expectations_swapped = [] 

    for pt_idx, pt_circuit_variant in enumerate(pt_variants_of_zne_circuit):
        ddd_variants_of_pt_circuit = ddd.construct_circuits(
            pt_circuit_variant, 
            rule=ddd.rules.xyxy
        )
        print(
            f"    DDD: Generated {len(ddd_variants_of_pt_circuit)} circuits "
            f"for PT variant {pt_idx+1}"
        )

        ddd_level_rem_corrected_measurements = [] 

        for ddd_idx, ddd_circuit_variant in enumerate(ddd_variants_of_pt_circuit):
            raw_measurement = noisy_exec(ddd_circuit_variant)
            
            rem_corrected_measurement = rem.mitigate_measurements(
                raw_measurement,
                inverse_confusion_matrix
            )
            ddd_level_rem_corrected_measurements.append(rem_corrected_measurement)
            
        exp_val_after_ddd_rem = obs._expectation_from_measurements(
            ddd_level_rem_corrected_measurements
        ).real
        pt_level_expectations_swapped.append(exp_val_after_ddd_rem)
        
    exp_val_for_this_sf_swapped = np.mean(pt_level_expectations_swapped)
    all_results_zne_level_swapped.append(exp_val_for_this_sf_swapped)
    print(
        f"  Scale factor {scale_factor_swapped} expectation "
        f"(avg over PT(avg over DDD(REM))): {exp_val_for_this_sf_swapped:.6f}"
    )

full_pipeline_result_val_swapped_order = zne.combine_results(
    scale_factors,
    all_results_zne_level_swapped,
    extrapolation_method=LinearFactory.extrapolate
)
if hasattr(full_pipeline_result_val_swapped_order, 'real'):
    full_pipeline_result_val_swapped_order = full_pipeline_result_val_swapped_order.real

print(f"{'='*70}")
print(
    f"\nFull pipeline (Swapped ZNE→PT→DDD→REM) result: "
    f"{full_pipeline_result_val_swapped_order:.6f}"
)
print(
    f"Full pipeline (Swapped ZNE→PT→DDD→REM) absolute error: "
    f"{abs(ideal_result_val - full_pipeline_result_val_swapped_order):.6f}"
)
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
    "REM+ZNE Pipeline": rem_zne_pipeline_result_val, 
    "Full Pipeline": full_pipeline_result_val,
    "Full Pipeline (ZNE→PT→DDD→REM)": full_pipeline_result_val_swapped_order
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
values_for_plot = [
    v.real if hasattr(v, 'real') else float(v) for v in results_summary.values()
]
errors_viz = [abs(ideal_result_val - val) for val in values_for_plot]

x_pos = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(12, 7))

color_error = 'salmon'
ax1.set_xlabel('Mitigation Strategy', fontsize=12)
ax1.set_ylabel('Absolute Error (from Ideal)', color=color_error, fontsize=12)
bars_error = ax1.bar(
    x_pos, 
    errors_viz, 
    width=0.6, 
    label='Absolute Error', 
    color=color_error, 
    alpha=0.7
)
ax1.tick_params(axis='y', labelcolor=color_error, labelsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

for bar_idx, bar in enumerate(bars_error):
    yval = bar.get_height()
    offset = 0.001 
    plt.text(
        bar.get_x() + bar.get_width()/2.0, 
        yval + offset, 
        f'{yval:.4f}', 
        ha='center', 
        va='bottom', 
        fontsize=9, 
        color='black'
    )

fig.tight_layout()
plt.title('Error Mitigation Pipeline: Absolute Error by Mitigation Strategy', fontsize=14)
ax1.legend(loc='upper right', fontsize=10) 

plt.show()
```

## Conclusion

This tutorial demonstrated how to construct an advanced error mitigation pipeline by combining Pauli Twirling (PT), Digital Dynamical Decoupling (DDD), Readout Error Mitigation (REM), and Zero-Noise Extrapolation (ZNE).

Key takeaways:
*   **Scale Factor Selection**: The choice of ZNE scale factors significantly impacts results. Our experiments show that smaller, more closely spaced factors ([1, 1.5, 2] versus [1, 2, 3]) provide better error reduction for this noise model.

*   **Simpler Can Be Better**: Surprisingly, the simpler REM+ZNE combination consistently outperforms more complex full pipelines. Adding more techniques doesn't always yield better results and may even be counterproductive if techniques interfere with each other.

*   **Technique Order Matters**: The order in which techniques are applied affects outcomes. When building full pipelines, applying PT before DDD (ZNE→PT→DDD→REM) performed better than applying DDD before PT in our experiments.

*   **Run-to-Run Variability**: Due to the stochastic nature of quantum noise and finite measurement shots, results can vary between runs. Users might observe different relative performance of techniques when executing this notebook.

*   **Technique-Specific Effects**: In our experiments, PT alone sometimes performed worse than the unmitigated case, suggesting it may convert noise into forms that more strongly affect this particular observable.

*   **API Pattern Consistency**: For DDD, REM, and ZNE, we followed a consistent pattern of first constructing necessary circuits or models (`construct_circuits`), then executing them, and finally combining the results (`combine_results`).

*   **Implementation Complexity vs. Benefit**: The full pipeline requires significantly more circuit executions. Users should evaluate whether this computational overhead is justified by any potential accuracy improvements, especially since simpler combinations like REM+ZNE may provide better results with fewer resources.

This tutorial provides a framework for experimenting with combined error mitigation approaches in Mitiq. For optimal results in real applications, it's recommended to first characterize the noise on your quantum hardware, then strategically select and combine the most effective techniques for your specific noise profile, and carefully tune parameters like ZNE scale factors.

```{code-cell} ipython3
# Display Mitiq version information
mitiq.about()
```