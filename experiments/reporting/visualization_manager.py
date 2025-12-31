"""
Visualization Manager
======================

Orchestrate creation of all experiment visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple

# Add project root to path
base_path = Path(__file__).parent.parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

from src.visualization.conceptual_diagrams import plot_prior_tempering_concept
from src.visualization.spatial_maps import (
    plot_spatial_comparison,
    generate_example_spatial_data
)
from src.visualization.training_diagnostics import (
    plot_transfer_gain_loss,
    plot_transfer_weight_evolution
)


class VisualizationManager:
    """Manage creation of all experiment visualizations."""

    def __init__(self, output_dir: Path, timestamp: str):
        """
        Initialize visualization manager.

        Args:
            output_dir: Directory to save visualizations
            timestamp: Timestamp for file naming
        """
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Custom color palette (colorblind-friendly)
        self.colors = {
            'fusion': '#0173B2',      # Blue
            'gam': '#DE8F05',         # Orange
            'positive': '#029E73',    # Green
            'negative': '#CC78BC',    # Purple
            'neutral': '#949494'      # Gray
        }

    def create_all(self, all_results: Dict, test_data: Dict = None):
        """
        Create all visualizations for the experiment.

        Args:
            all_results: Complete results dictionary
            test_data: Test data for spatial visualizations (optional)
        """
        print(f"\n🎨 Creating visualizations...")

        # Create transfer learning visualizations
        self.create_transfer_visualizations(all_results)

        # Create conceptual and diagnostic visualizations
        self.create_conceptual_visualizations(all_results, test_data)

        print(f"\n✅ All visualizations saved to: {self.fig_dir.relative_to(self.output_dir.parent)}/")

    def create_transfer_visualizations(self, all_results: Dict):
        """Create main transfer learning result visualizations."""
        # Extract data
        fusiongp_pt = all_results['fusiongp_prior_tempering']['results']
        gam_pt = all_results['gam_ssm_lur_prior_tempering']['results']
        fusiongp_obtl = all_results['fusiongp_obtl']['results']
        gam_obtl = all_results['gam_ssm_lur_obtl']['results']

        # Create Prior Tempering visualization
        self._create_prior_tempering_figure(fusiongp_pt, gam_pt)

        # Create OBTL visualization
        self._create_obtl_figure(fusiongp_obtl, gam_obtl)

        # Create summary comparison
        self._create_summary_figure(all_results)

        # Create R² distribution visualization
        self._create_r2_distribution_figure(all_results)

        # Create R² heatmap visualization
        self._create_r2_heatmap(all_results)

    def create_conceptual_visualizations(self, all_results: Dict, test_data: Dict = None):
        """Create conceptual diagrams and training diagnostics."""
        # Image 1: Prior Tempering Concept
        try:
            concept_path = self.fig_dir / f'prior_tempering_concept_{self.timestamp}.png'
            plot_prior_tempering_concept(output_path=concept_path, save_pdf=True)
        except Exception as e:
            print(f"   ⚠️  Could not create prior tempering concept: {e}")

        # Image 3: Spatial Comparison Maps
        try:
            predictions, uncertainties, coords = generate_example_spatial_data(
                n_locations=50, seed=42
            )
            spatial_path = self.fig_dir / f'spatial_comparison_{self.timestamp}.png'
            plot_spatial_comparison(
                predictions=predictions,
                uncertainties=uncertainties,
                coords=coords,
                output_path=spatial_path,
                save_pdf=True
            )
        except Exception as e:
            print(f"   ⚠️  Could not create spatial comparison: {e}")

        # Training Diagnostics: Transfer Gain/Loss
        self._create_gain_loss_diagnostics(all_results)

        # Training Diagnostics: OBTL Weight Evolution
        self._create_weight_evolution_diagnostic(all_results)

    def _create_prior_tempering_figure(self, fusiongp_pt, gam_pt):
        """Create Prior Tempering multi-panel figure."""
        # Prepare DataFrames
        pt_data = []
        for r in sorted(fusiongp_pt, key=lambda x: x['lambda']):
            pt_data.append({'Model': 'FusionGP', 'λ': r['lambda'], 'RMSE': r['rmse'],
                           'MAE': r['mae'], 'R²': r['r2']})
        for r in sorted(gam_pt, key=lambda x: x['lambda']):
            pt_data.append({'Model': 'GAM-SSM-LUR', 'λ': r['lambda'], 'RMSE': r['rmse'],
                           'MAE': r['mae'], 'R²': r['r2']})
        df_pt = pd.DataFrame(pt_data)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.patch.set_facecolor('white')
        fig.suptitle('Prior Tempering Transfer Learning Results',
                     fontsize=18, fontweight='bold', y=0.998)

        # (a) RMSE vs Lambda
        ax = axes[0, 0]
        for model, color in [('FusionGP', self.colors['fusion']), ('GAM-SSM-LUR', self.colors['gam'])]:
            data = df_pt[df_pt['Model'] == model]
            sns.lineplot(data=data, x='λ', y='RMSE', ax=ax,
                        marker='o', markersize=10, linewidth=3,
                        color=color, label=model)
            best_idx = data['RMSE'].idxmin()
            best_lambda = data.loc[best_idx, 'λ']
            best_rmse = data.loc[best_idx, 'RMSE']
            ax.scatter([best_lambda], [best_rmse], s=400, marker='*',
                      color='gold', edgecolor='black', linewidth=2, zorder=10)

        ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE (µg/m³)', fontsize=13, fontweight='bold')
        ax.set_title('(a) Prediction Error vs Temperature', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=12, frameon=True, shadow=True)
        sns.despine(ax=ax)

        # (b) MAE vs Lambda
        ax = axes[0, 1]
        for model, color in [('FusionGP', self.colors['fusion']), ('GAM-SSM-LUR', self.colors['gam'])]:
            data = df_pt[df_pt['Model'] == model]
            sns.lineplot(data=data, x='λ', y='MAE', ax=ax,
                        marker='s', markersize=10, linewidth=3,
                        color=color, label=model)

        ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAE (µg/m³)', fontsize=13, fontweight='bold')
        ax.set_title('(b) Mean Absolute Error vs Temperature', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=12, frameon=True, shadow=True)
        sns.despine(ax=ax)

        # (c) R² vs Lambda
        ax = axes[1, 0]
        for model, color in [('FusionGP', self.colors['fusion']), ('GAM-SSM-LUR', self.colors['gam'])]:
            data = df_pt[df_pt['Model'] == model]
            sns.lineplot(data=data, x='λ', y='R²', ax=ax,
                        marker='D', markersize=10, linewidth=3,
                        color=color, label=model)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Baseline (R²=0)')
        ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
        ax.set_title('(c) Explained Variance vs Temperature', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=12, frameon=True, shadow=True)
        sns.despine(ax=ax)

        # (d) Improvement over Baseline
        ax = axes[1, 1]
        baseline_fusion = df_pt[(df_pt['Model'] == 'FusionGP') & (df_pt['λ'] == 0.0)]['RMSE'].values[0]
        baseline_gam = df_pt[(df_pt['Model'] == 'GAM-SSM-LUR') & (df_pt['λ'] == 0.0)]['RMSE'].values[0]

        for model, baseline, color in [('FusionGP', baseline_fusion, self.colors['fusion']),
                                        ('GAM-SSM-LUR', baseline_gam, self.colors['gam'])]:
            data = df_pt[df_pt['Model'] == model].copy()
            data['Improvement (%)'] = (baseline - data['RMSE']) / baseline * 100
            sns.lineplot(data=data, x='λ', y='Improvement (%)', ax=ax,
                        marker='o', markersize=10, linewidth=3,
                        color=color, label=model)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        ax.fill_between([-0.1, 1.1], 0, 20, alpha=0.15, color=self.colors['positive'],
                       label='Positive Transfer')
        ax.fill_between([-0.1, 1.1], 0, -20, alpha=0.15, color=self.colors['negative'],
                       label='Negative Transfer')

        ax.set_xlabel('Temperature Parameter (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE Improvement (%)', fontsize=13, fontweight='bold')
        ax.set_title('(d) Transfer Learning Benefit', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, frameon=True, shadow=True, loc='best')
        ax.set_xlim(-0.05, 1.05)
        sns.despine(ax=ax)

        plt.tight_layout(rect=[0, 0, 1, 0.995])
        fig_path = self.fig_dir / f'prior_tempering_fancy_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {fig_path.name}")
        plt.close(fig)

    def _create_obtl_figure(self, fusiongp_obtl, gam_obtl):
        """Create OBTL results figure."""
        # Prepare DataFrames
        obtl_data = []
        for r in sorted(fusiongp_obtl, key=lambda x: x['delta']):
            obtl_data.append({'Model': 'FusionGP', 'δ': r['delta'], 'RMSE': r['rmse'],
                             'MAE': r['mae'], 'R²': r['r2']})
        for r in sorted(gam_obtl, key=lambda x: x['delta']):
            obtl_data.append({'Model': 'GAM-SSM-LUR', 'δ': r['delta'], 'RMSE': r['rmse'],
                             'MAE': r['mae'], 'R²': r['r2']})
        df_obtl = pd.DataFrame(obtl_data)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle('OBTL (Optimal Bayesian Transfer Learning) Results',
                     fontsize=18, fontweight='bold', y=1.00)

        # (a) RMSE vs Delta
        ax = axes[0]
        for model, color in [('FusionGP', self.colors['fusion']), ('GAM-SSM-LUR', self.colors['gam'])]:
            data = df_obtl[df_obtl['Model'] == model]
            sns.lineplot(data=data, x='δ', y='RMSE', ax=ax,
                        marker='o', markersize=12, linewidth=3.5,
                        color=color, label=model)
            best_idx = data['RMSE'].idxmin()
            best_delta = data.loc[best_idx, 'δ']
            best_rmse = data.loc[best_idx, 'RMSE']
            ax.scatter([best_delta], [best_rmse], s=500, marker='*',
                      color='gold', edgecolor='black', linewidth=2.5, zorder=10)

        ax.set_xlabel('Transfer Strength (δ)', fontsize=14, fontweight='bold')
        ax.set_ylabel('RMSE (µg/m³)', fontsize=14, fontweight='bold')
        ax.set_title('(a) Prediction Error vs Transfer Strength', fontsize=15, fontweight='bold', pad=12)
        ax.legend(fontsize=13, frameon=True, shadow=True)
        sns.despine(ax=ax)

        # (b) R² vs Delta
        ax = axes[1]
        for model, color in [('FusionGP', self.colors['fusion']), ('GAM-SSM-LUR', self.colors['gam'])]:
            data = df_obtl[df_obtl['Model'] == model]
            sns.lineplot(data=data, x='δ', y='R²', ax=ax,
                        marker='D', markersize=12, linewidth=3.5,
                        color=color, label=model)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label='Baseline (R²=0)')
        ax.set_xlabel('Transfer Strength (δ)', fontsize=14, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
        ax.set_title('(b) Explained Variance vs Transfer Strength', fontsize=15, fontweight='bold', pad=12)
        ax.legend(fontsize=13, frameon=True, shadow=True)
        sns.despine(ax=ax)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig_path = self.fig_dir / f'obtl_fancy_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {fig_path.name}")
        plt.close(fig)

    def _create_summary_figure(self, all_results: Dict):
        """Create summary comparison figure."""
        # Extract best results
        best_results = {
            'FusionGP\nPT': all_results['fusiongp_prior_tempering']['best'],
            'FusionGP\nOBTL': all_results['fusiongp_obtl']['best'],
            'GAM-SSM-LUR\nPT': all_results['gam_ssm_lur_prior_tempering']['best'],
            'GAM-SSM-LUR\nOBTL': all_results['gam_ssm_lur_obtl']['best']
        }

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')

        labels = list(best_results.keys())
        rmse_values = [best_results[k]['rmse'] for k in labels]
        bar_colors = [self.colors['fusion'], self.colors['fusion'],
                      self.colors['gam'], self.colors['gam']]

        bars = ax.bar(labels, rmse_values, color=bar_colors, edgecolor='black',
                      linewidth=2, alpha=0.8, width=0.6)

        # Add value labels
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.2f}', ha='center', va='bottom',
                   fontsize=13, fontweight='bold')

        ax.set_ylabel('Best RMSE (µg/m³)', fontsize=14, fontweight='bold')
        ax.set_title('Best Transfer Learning Results Comparison',
                    fontsize=16, fontweight='bold', pad=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        sns.despine(ax=ax)

        plt.tight_layout()
        fig_path = self.fig_dir / f'summary_comparison_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {fig_path.name}")
        plt.close(fig)

    def _create_gain_loss_diagnostics(self, all_results: Dict):
        """Create gain/loss waterfall charts."""
        try:
            # Prior Tempering Gain/Loss - FusionGP
            pt_path = self.fig_dir / f'pt_gain_loss_{self.timestamp}.png'
            plot_transfer_gain_loss(
                results=all_results['fusiongp_prior_tempering'],
                baseline_metric=0.92,  # Baseline RMSE
                output_path=pt_path,
                save_pdf=True,
                metric='rmse'
            )

            # OBTL Gain/Loss - FusionGP
            obtl_path = self.fig_dir / f'obtl_gain_loss_{self.timestamp}.png'
            plot_transfer_gain_loss(
                results=all_results['fusiongp_obtl'],
                baseline_metric=0.92,  # Baseline RMSE
                output_path=obtl_path,
                save_pdf=True,
                metric='rmse'
            )
        except Exception as e:
            print(f"   ⚠️  Could not create gain/loss diagnostics: {e}")

    def _create_weight_evolution_diagnostic(self, all_results: Dict):
        """Create OBTL weight evolution visualization."""
        try:
            weight_path = self.fig_dir / f'obtl_weights_{self.timestamp}.png'
            # Extract results arrays from both models
            fgp_results = all_results['fusiongp_obtl'].get('results', [])
            gam_results = all_results['gam_ssm_lur_obtl'].get('results', [])

            # Combine all OBTL results for weight evolution plot
            obtl_results = fgp_results + gam_results

            if obtl_results:
                plot_transfer_weight_evolution(
                    obtl_results=obtl_results,
                    output_path=weight_path,
                    save_pdf=True
                )
        except Exception as e:
            print(f"   ⚠️  Could not create weight evolution: {e}")

    def _create_r2_distribution_figure(self, all_results: Dict):
        """Create R² distribution visualization showing all configurations."""
        # Collect all results
        fgp_pt = sorted(all_results['fusiongp_prior_tempering']['results'],
                       key=lambda x: x['lambda'])
        gam_pt = sorted(all_results['gam_ssm_lur_prior_tempering']['results'],
                       key=lambda x: x['lambda'])
        fgp_obtl = sorted(all_results['fusiongp_obtl']['results'],
                         key=lambda x: x['delta'])
        gam_obtl = sorted(all_results['gam_ssm_lur_obtl']['results'],
                         key=lambda x: x['delta'])
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Subplot 1: Prior Tempering R² Distribution ---
        ax1 = axes[0]
        
        lambdas = [r['lambda'] for r in fgp_pt]
        fgp_r2 = [r['r2'] for r in fgp_pt]
        gam_r2 = [r['r2'] for r in gam_pt]
        
        x_pos = np.arange(len(lambdas))
        width = 0.35
        
        # Plot bars
        bars1 = ax1.bar(x_pos - width/2, fgp_r2, width, 
                       label='FusionGP', color=self.colors['fusion'], alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, gam_r2, width,
                       label='GAM-SSM-LUR', color=self.colors['gam'], alpha=0.8)
        
        # Highlight positive R²
        for i, (r2_f, r2_g) in enumerate(zip(fgp_r2, gam_r2)):
            if r2_f > 0:
                bars1[i].set_edgecolor('green')
                bars1[i].set_linewidth(3)
            if r2_g > 0:
                bars2[i].set_edgecolor('green')
                bars2[i].set_linewidth(3)
        
        # Add zero line
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Add trend lines
        ax1.plot(x_pos, fgp_r2, color=self.colors['fusion'], linestyle='-',
                linewidth=2, alpha=0.6, marker='o', markersize=4)
        ax1.plot(x_pos, gam_r2, color=self.colors['gam'], linestyle='-',
                linewidth=2, alpha=0.6, marker='s', markersize=4)

        ax1.set_xlabel(r'Prior Tempering Parameter $\lambda$', fontsize=12)
        ax1.set_ylabel(r'$R^2$ Score', fontsize=12)
        ax1.set_title('(a) Prior Tempering: Only λ=0 Achieves Positive R²', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{lam:.1f}' for lam in lambdas])
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)

        # --- Subplot 2: OBTL R² Distribution ---
        ax2 = axes[1]
        
        deltas = [r['delta'] for r in fgp_obtl]
        fgp_obtl_r2 = [r['r2'] for r in fgp_obtl]
        gam_obtl_r2 = [r['r2'] for r in gam_obtl]
        
        x_pos2 = np.arange(len(deltas))
        
        # Plot bars
        ax2.bar(x_pos2 - width/2, fgp_obtl_r2, width,
               label='FusionGP', color=self.colors['fusion'], alpha=0.8)
        ax2.bar(x_pos2 + width/2, gam_obtl_r2, width,
               label='GAM-SSM-LUR', color=self.colors['gam'], alpha=0.8)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Add trend lines
        ax2.plot(x_pos2, fgp_obtl_r2, color=self.colors['fusion'], linestyle='-',
                linewidth=2, alpha=0.6, marker='o', markersize=4)
        ax2.plot(x_pos2, gam_obtl_r2, color=self.colors['gam'], linestyle='-',
                linewidth=2, alpha=0.6, marker='s', markersize=4)

        ax2.set_xlabel(r'OBTL Concentration Parameter $\delta$', fontsize=12)
        ax2.set_ylabel(r'$R^2$ Score', fontsize=12)
        ax2.set_title('(b) OBTL: Wide Range of Negative R² Values', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels([f'{d:.1f}' for d in deltas])
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.fig_dir / f'r2_distribution_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()

        print(f"   ✓ R² distribution: {output_path.name}")

    def _create_r2_heatmap(self, all_results: Dict):
        """Create comprehensive R² heatmap showing all 18 configurations."""
        # Extract R² values for all configurations
        # Prior Tempering: λ ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
        fgp_pt = sorted(all_results['fusiongp_prior_tempering']['results'],
                       key=lambda x: x['lambda'])
        gam_pt = sorted(all_results['gam_ssm_lur_prior_tempering']['results'],
                       key=lambda x: x['lambda'])

        # OBTL: δ ∈ {0.3, 0.5, 0.7, 1.0}
        fgp_obtl = sorted(all_results['fusiongp_obtl']['results'],
                         key=lambda x: x['delta'])
        gam_obtl = sorted(all_results['gam_ssm_lur_obtl']['results'],
                         key=lambda x: x['delta'])

        # Combine into rows (FusionGP, GAM-SSM-LUR)
        fgp_row = [r['r2'] for r in fgp_pt] + [r['r2'] for r in fgp_obtl]
        gam_row = [r['r2'] for r in gam_pt] + [r['r2'] for r in gam_obtl]

        # Create 2D array
        data = np.array([fgp_row, gam_row])

        # Labels
        row_labels = ['FusionGP', 'GAM-SSM-LUR']
        col_labels = [f'PT λ={r["lambda"]:.1f}' for r in fgp_pt] + \
                     [f'OBTL δ={r["delta"]:.1f}' for r in fgp_obtl]

        # Create figure with extra space at top
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('white')

        # Custom colormap for severity levels
        colors_list = [
            (0.0, 'darkred'),      # Catastrophic: < -100
            (0.3, 'red'),          # Very bad: -100 to -10
            (0.6, 'orange'),       # Bad: -10 to 0
            (0.8, 'yellow'),       # Slightly negative
            (1.0, 'lightgreen')    # Positive
        ]
        cmap = LinearSegmentedColormap.from_list('custom', colors_list)

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-50, vmax=5)

        # Set ticks
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=10, rotation=45, ha='right')
        ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')

        # Add title with extra padding
        ax.set_title('Complete R² Distribution Across All 18 Model Configurations\n(Only 2 configurations achieve positive R²)',
                     fontsize=13, fontweight='bold', pad=60)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('R² Value', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

        # Add grid
        ax.set_xticks(np.arange(len(col_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5)

        # Add vertical separator between PT and OBTL
        ax.axvline(x=4.5, color='black', linewidth=3, linestyle='--')

        # Annotate cells with R² values
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = data[i, j]
                # Choose text color based on background
                text_color = 'white' if value < -10 else 'black'

                # Format text based on magnitude
                if abs(value) > 100:
                    text = f'{value:.0f}'
                else:
                    text = f'{value:.1f}'

                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontweight='bold', fontsize=9)

        # Add method labels at the top
        ax.text(2, -1.0, 'Prior Tempering', ha='center', va='center',
               fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
               facecolor='lightblue', edgecolor='blue', linewidth=2))
        ax.text(6.5, -1.0, 'OBTL', ha='center', va='center',
               fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
               facecolor='lightcoral', edgecolor='red', linewidth=2))

        # Adjust layout with space at top for labels
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at top for method labels

        # Save figure
        png_path = self.fig_dir / f'r2_heatmap_{self.timestamp}.png'
        pdf_path = self.fig_dir / f'r2_heatmap_{self.timestamp}.pdf'

        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.close()

        # Summary statistics
        positive_count = np.sum(data > 0)
        total_count = data.size

        print(f"   ✓ R² heatmap: {png_path.name} ({positive_count}/{total_count} positive)")
