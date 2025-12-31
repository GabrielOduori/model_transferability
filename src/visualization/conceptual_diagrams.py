"""
Conceptual diagram generators for transfer learning methodology.

This module creates publication-quality conceptual diagrams to illustrate
Bayesian transfer learning concepts, particularly prior tempering.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from scipy.stats import norm


def plot_prior_tempering_concept(
    output_path: Optional[Path] = None,
    save_pdf: bool = True
) -> Path:
    """
    Create conceptual diagram showing Bayesian Prior Tempering.

    Generates a figure with three Gaussian curves illustrating variance inflation:
    - Source Posterior (β=1.0): Tight distribution from source domain
    - Tempered Prior (β=0.5): Inflated variance for target domain
    - Non-informative (β→0): Nearly uniform prior

    Args:
        output_path: Path to save the figure. If None, saves to current directory.
        save_pdf: If True, also save as PDF for LaTeX inclusion.

    Returns:
        Path to the saved PNG file.

    Example:
        >>> from pathlib import Path
        >>> output_dir = Path('results/experiment_20241228_154530/figures')
        >>> plot_prior_tempering_concept(output_dir / 'prior_tempering_concept.png')
    """
    # Set up publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'serif',
        'text.usetex': False  # Set to True if LaTeX is available
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Define x-axis range
    x = np.linspace(-10, 10, 1000)

    # Define three distributions with increasing variance
    # All centered at μ=0 for simplicity

    # 1. Source Posterior (β=1.0): σ = 1.0
    mu_source = 0
    sigma_source = 1.0
    y_source = norm.pdf(x, mu_source, sigma_source)

    # 2. Tempered Prior (β=0.5): σ = 2.0 (variance inflated by 1/β)
    mu_tempered = 0
    sigma_tempered = 2.0
    y_tempered = norm.pdf(x, mu_tempered, sigma_tempered)

    # 3. Non-informative (β→0): σ = 5.0 (very flat)
    mu_noninformative = 0
    sigma_noninformative = 5.0
    y_noninformative = norm.pdf(x, mu_noninformative, sigma_noninformative)

    # Plot distributions
    ax.plot(x, y_source, 'b-', linewidth=2.5, label='Source Posterior (β=1.0)', zorder=3)
    ax.plot(x, y_tempered, 'g--', linewidth=2.5, label='Tempered Prior (β=0.5)', zorder=2)
    ax.plot(x, y_noninformative, 'r:', linewidth=2.5, label='Non-informative (β→0)', zorder=1)

    # Fill areas for better visualization
    ax.fill_between(x, y_source, alpha=0.2, color='blue')
    ax.fill_between(x, y_tempered, alpha=0.15, color='green')
    ax.fill_between(x, y_noninformative, alpha=0.1, color='red')

    # Add arrow showing variance inflation
    arrow_y = 0.35  # Position arrow above distributions
    arrow_start_x = -2
    arrow_end_x = 5

    ax.annotate(
        '',
        xy=(arrow_end_x, arrow_y),
        xytext=(arrow_start_x, arrow_y),
        arrowprops=dict(
            arrowstyle='->',
            lw=2,
            color='black',
            connectionstyle='arc3,rad=0'
        )
    )

    # Add arrow label
    ax.text(
        (arrow_start_x + arrow_end_x) / 2,
        arrow_y + 0.02,
        'Increasing Temperature\n(Variance Inflation)',
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold'
    )

    # Styling
    ax.set_xlabel('Parameter Value', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Bayesian Prior Tempering for Transfer Learning',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.45)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = Path('prior_tempering_concept.png')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    # Save PDF if requested (for LaTeX inclusion)
    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   ✓ Saved: {pdf_path.name}")

    plt.close()

    return output_path


if __name__ == '__main__':
    # Test the function
    print("Generating Prior Tempering conceptual diagram...")
    output_path = plot_prior_tempering_concept()
    print(f"Diagram saved to: {output_path}")
