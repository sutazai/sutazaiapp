"""Visualization of SutazAi Coherence Preservation Techniques."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sutazai_core.neural_entanglement.coherence_preserver import (
    CoherencePreserver,
    ErrorMitigationStrategy,
)


def visualize_error_mitigation(
    state: np.ndarray,
    strategy: ErrorMitigationStrategy,
    coherence_preserver: CoherencePreserver,
):
    """
    Visualize the error mitigation process for a given strategy.

    Args:
        state (np.ndarray): Input noisy state.
        strategy (ErrorMitigationStrategy): Error mitigation strategy.
        coherence_preserver (CoherencePreserver): Coherence preservation instance.
    """
    # Perform error mitigation
    mitigated_state, metrics = coherence_preserver.sutazai_error_mitigation(
        state, strategy=strategy
    )

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original State
    plt.subplot(1, 3, 1)
    sns.heatmap(state, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar=False)
    plt.title(f"Original Noisy State\nNoise Level: {metrics.noise_level:.4f}")

    # Mitigated State
    plt.subplot(1, 3, 2)
    sns.heatmap(
        mitigated_state,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        cbar=False,
    )
    plt.title(f"Mitigated State\nStrategy: {strategy.name}")

    # Difference Heatmap
    plt.subplot(1, 3, 3)
    difference = mitigated_state - state
    sns.heatmap(difference, cmap="RdBu_r", center=0, annot=True, fmt=".4f", cbar=False)
    plt.title(
        f"Correction Difference\nEntropy Reduction: {metrics.entropy_reduction:.4f}"
    )

    plt.tight_layout()
    plt.suptitle(f"Coherence Preservation: {strategy.name}", fontsize=16)
    plt.show()


def main():
    """Demonstrate visualization of coherence preservation techniques."""
    # Configure coherence preserver
    coherence_preserver = CoherencePreserver(
        precision=0.999, logging_level=logging.INFO, max_history=50
    )

    # Generate sample noisy states with different characteristics
    noisy_states = [
        np.random.normal(0, 1, (5, 5)),  # High noise
        np.random.normal(0, 0.5, (5, 5)),  # Medium noise
        np.random.normal(0, 0.1, (5, 5)),  # Low noise
    ]

    # Strategies to visualize
    strategies = [
        ErrorMitigationStrategy.BASIC_CORRECTION,
        ErrorMitigationStrategy.ADAPTIVE_FILTERING,
        ErrorMitigationStrategy.PROBABILISTIC_RECOVERY,
        ErrorMitigationStrategy.MACHINE_LEARNING_CORRECTION,
        ErrorMitigationStrategy.ADVANCED_ML_CORRECTION,
    ]

    # Visualize error mitigation for each noise level and strategy
    for noise_level, noisy_state in enumerate(noisy_states):
        print(f"\nNoise Level: {['High', 'Medium', 'Low'][noise_level]}")

        for strategy in strategies:
            print(f"\nStrategy: {strategy.name}")
            visualize_error_mitigation(noisy_state, strategy, coherence_preserver)


if __name__ == "__main__":
    main()
