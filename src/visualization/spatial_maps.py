"""
Spatial comparison map generators for transfer learning evaluation.

This module creates side-by-side spatial maps comparing standard transfer
(crisp predictions) with probabilistic transfer (predictions + uncertainty).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def plot_spatial_comparison(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    coords: np.ndarray,
    output_path: Optional[Path] = None,
    save_pdf: bool = True,
    title: str = "Standard vs Probabilistic Transfer Learning"
) -> Path:
    """
    Create side-by-side spatial comparison maps.

    Left panel: Standard transfer with crisp predictions (no uncertainty)
    Right panel: Probabilistic transfer with predictions + uncertainty overlay

    Args:
        predictions: Array of PM2.5 predictions (µg/m³), shape (n_locations,)
        uncertainties: Array of prediction uncertainties (std dev), shape (n_locations,)
        coords: Array of coordinates [longitude, latitude], shape (n_locations, 2)
        output_path: Path to save the figure. If None, saves to current directory.
        save_pdf: If True, also save as PDF for LaTeX inclusion.
        title: Main title for the figure.

    Returns:
        Path to the saved PNG file.

    Example:
        >>> predictions = np.array([8.5, 12.3, 15.7, 9.2, 11.8])
        >>> uncertainties = np.array([1.2, 2.5, 1.8, 1.5, 2.0])
        >>> coords = np.array([[-0.1, 51.5], [-0.15, 51.52], ...])
        >>> plot_spatial_comparison(predictions, uncertainties, coords)
    """
    # Set up publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'font.family': 'sans-serif'
    })

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Define air quality categories (UK standards for PM2.5)
    def categorize_pm25(values):
        """Categorize PM2.5 values into Good/Moderate/Poor."""
        categories = np.zeros_like(values, dtype=int)
        categories[values <= 12] = 0  # Good
        categories[(values > 12) & (values <= 35)] = 1  # Moderate
        categories[values > 35] = 2  # Poor
        return categories

    # Categorize predictions
    categories = categorize_pm25(predictions)

    # Define colormap for air quality
    colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
    n_bins = 3
    cmap = LinearSegmentedColormap.from_list('air_quality', colors, N=n_bins)

    # ====================
    # LEFT PANEL: Standard Transfer (crisp predictions only)
    # ====================
    ax_left = axes[0]

    # Create scatter plot with category colors
    scatter_left = ax_left.scatter(
        coords[:, 0],
        coords[:, 1],
        c=predictions,
        cmap=cmap,
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        vmin=0,
        vmax=40
    )

    # Add colorbar
    cbar_left = plt.colorbar(scatter_left, ax=ax_left, pad=0.02)
    cbar_left.set_label('PM$_{2.5}$ (µg/m³)', fontweight='bold')

    # Styling
    ax_left.set_xlabel('Longitude', fontweight='bold')
    ax_left.set_ylabel('Latitude', fontweight='bold')
    ax_left.set_title('Standard Transfer\n(No Uncertainty)', fontweight='bold', pad=10)
    ax_left.grid(True, alpha=0.3, linestyle='--')

    # Add category legend
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Good (≤12)'),
        mpatches.Patch(facecolor='#FFC107', edgecolor='black', label='Moderate (12-35)'),
        mpatches.Patch(facecolor='#F44336', edgecolor='black', label='Poor (>35)')
    ]
    ax_left.legend(handles=legend_elements, loc='upper right', title='Air Quality')

    # ====================
    # RIGHT PANEL: Probabilistic Transfer (predictions + uncertainty)
    # ====================
    ax_right = axes[1]

    # Create scatter plot with predictions
    scatter_right = ax_right.scatter(
        coords[:, 0],
        coords[:, 1],
        c=predictions,
        cmap=cmap,
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        vmin=0,
        vmax=40
    )

    # Overlay uncertainty as error circles
    # Size of circle represents uncertainty magnitude
    for i in range(len(coords)):
        circle = plt.Circle(
            (coords[i, 0], coords[i, 1]),
            radius=uncertainties[i] * 0.002,  # Scale for visibility
            color='blue',
            alpha=0.2,
            fill=True,
            linestyle='--',
            linewidth=1.5
        )
        ax_right.add_patch(circle)

    # Add colorbar
    cbar_right = plt.colorbar(scatter_right, ax=ax_right, pad=0.02)
    cbar_right.set_label('PM$_{2.5}$ (µg/m³)', fontweight='bold')

    # Styling
    ax_right.set_xlabel('Longitude', fontweight='bold')
    ax_right.set_ylabel('Latitude', fontweight='bold')
    ax_right.set_title('Probabilistic Transfer\n(With Uncertainty)', fontweight='bold', pad=10)
    ax_right.grid(True, alpha=0.3, linestyle='--')

    # Add category legend
    ax_right.legend(handles=legend_elements, loc='upper right', title='Air Quality')

    # Add uncertainty indicator
    ax_right.text(
        0.02, 0.02,
        'Blue circles = Uncertainty\n(95% confidence)',
        transform=ax_right.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = Path('spatial_comparison.png')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path.name}")

    # Save PDF if requested
    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"   ✓ Saved: {pdf_path.name}")

    plt.close()

    return output_path


def generate_example_spatial_data(
    n_locations: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate example spatial data for testing.

    Args:
        n_locations: Number of spatial locations to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (predictions, uncertainties, coords)
    """
    np.random.seed(seed)

    # Generate random coordinates (roughly London area)
    lon = np.random.uniform(-0.3, 0.1, n_locations)
    lat = np.random.uniform(51.4, 51.6, n_locations)
    coords = np.column_stack([lon, lat])

    # Generate predictions with spatial structure
    # Higher pollution in city center
    center = np.array([-0.1, 51.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    predictions = 25 - distances * 30 + np.random.normal(0, 2, n_locations)
    predictions = np.clip(predictions, 5, 40)

    # Generate uncertainties (higher uncertainty farther from training data)
    uncertainties = 1.5 + distances * 5 + np.random.uniform(0, 1, n_locations)

    return predictions, uncertainties, coords


if __name__ == '__main__':
    # Test the function with example data
    print("Generating spatial comparison maps...")

    predictions, uncertainties, coords = generate_example_spatial_data()
    output_path = plot_spatial_comparison(
        predictions,
        uncertainties,
        coords,
        title="Standard vs Probabilistic Transfer Learning (Example)"
    )

    print(f"Maps saved to: {output_path}")
