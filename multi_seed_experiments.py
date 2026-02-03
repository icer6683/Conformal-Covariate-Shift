#!/usr/bin/env python3
"""
=============================================================================
MULTI-SEED CONFORMAL COVERAGE EXPERIMENTS
=============================================================================

Wrapper to run test_conformal.py experiments across multiple seeds and
aggregate results for statistical robustness.

USAGE:
  python multi_seed_experiments.py --predictor algorithm --n_seeds 100
  python multi_seed_experiments.py --predictor adaptive --with_shift --n_seeds 50
  
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # For progress bars (pip install tqdm)

from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor
from adaptive_conformal import OnlineConformalPredictor
from algorithm import AdaptedCAFHT
from test_conformal import run_time_based_coverage_experiment, GAMMA_GRID


class MultiSeedExperiment:
    """Run and aggregate conformal prediction experiments across multiple seeds."""
    
    def __init__(self, base_config: dict, n_seeds: int = 100, base_seed: int = 1000):
        """
        Initialize multi-seed experiment runner.
        
        Args:
            base_config: Dictionary of experiment configuration parameters
            n_seeds: Number of different seeds to run
            base_seed: Starting seed value (will use base_seed, base_seed+1, ...)
        """
        self.base_config = base_config
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.results_by_seed = {}
        
    def run_all_seeds(self) -> Dict:
        """Run experiments for all seeds and collect results."""
        print(f"Running {self.n_seeds} experiments with different seeds...")
        print(f"Base configuration: {self.base_config['predictor']} predictor, "
              f"{'with' if self.base_config['with_shift'] else 'no'} shift")
        print("=" * 70)
        
        for seed_idx in tqdm(range(self.n_seeds), desc="Running seeds"):
            current_seed = self.base_seed + seed_idx
            
            # Run single seed experiment
            results = self._run_single_seed(current_seed)
            self.results_by_seed[current_seed] = results
            
            # Periodic progress update
            if (seed_idx + 1) % 10 == 0:
                self._print_progress_summary(seed_idx + 1)
        
        print("\nAll seeds completed!")
        return self.results_by_seed
    
    def _run_single_seed(self, seed: int) -> Dict:
        """Run experiment for a single seed."""
        config = self.base_config.copy()
        config['seed'] = seed
        
        # Initialize generator
        generator = TimeSeriesGenerator(T=config['T'], d=1, seed=seed)
        
        # Initialize predictor
        predictor = self._create_predictor(config)
        
        # Run experiment
        results_by_time = run_time_based_coverage_experiment(
            generator=generator,
            predictor=predictor,
            predictor_type=config['predictor'],
            n_series=config['n_series'],
            covariate_mode=config['covariate_mode'],
            ar_coef=config['ar_coef'],
            beta=config['beta'],
            noise_std=config['noise_std'],
            trend_coef=config['trend_coef'],
            covar_rate=config['covar_rate'],
            covar_rate_shift=config['covar_rate_shift'],
            x_rate=config['x_rate'],
            x_trend=config['x_trend'],
            x_noise_std=config['x_noise_std'],
            x0_lambda=config['x0_lambda'],
            x_rate_shift=config['x_rate_shift'],
            x_trend_shift=config['x_trend_shift'],
            x_noise_std_shift=config['x_noise_std_shift'],
            x0_lambda_shift=config['x0_lambda_shift'],
            with_shift=config['with_shift'],
            n_train=config['n_train'],
            n_cal=config['n_cal'],
            aci_stepsize=config['aci_stepsize'],
        )
        
        return results_by_time
    
    def _create_predictor(self, config: dict):
        """Create predictor based on configuration."""
        if config['predictor'] == 'basic':
            return BasicConformalPredictor(alpha=config['alpha'])
        elif config['predictor'] == 'adaptive':
            return OnlineConformalPredictor(
                alpha=config['alpha'],
                window_size=config.get('window_size', 800)
            )
        else:  # algorithm
            return AdaptedCAFHT(alpha=config['alpha'])
    
    def _print_progress_summary(self, n_completed: int):
        """Print summary statistics for completed runs."""
        seeds_completed = sorted(list(self.results_by_seed.keys()))[:n_completed]
        
        # Calculate overall coverage across all completed seeds
        all_coverage = []
        for seed in seeds_completed:
            results = self.results_by_seed[seed]
            for key in results:
                if isinstance(key, int):  # Time step
                    all_coverage.extend(results[key]['coverage_history'])
        
        mean_cov = np.mean(all_coverage)
        target = 1 - self.base_config['alpha']
        
        print(f"\n  After {n_completed} seeds: "
              f"Mean coverage = {mean_cov:.3f} (target = {target:.3f})")
    
    def aggregate_results(self) -> Dict:
        """
        Aggregate results across all seeds.
        
        Returns:
            Dictionary with aggregated statistics by time step
        """
        print("\nAggregating results across all seeds...")
        
        # Get all time steps (should be same across all seeds)
        first_seed = list(self.results_by_seed.keys())[0]
        time_steps = sorted([k for k in self.results_by_seed[first_seed].keys() 
                           if isinstance(k, int)])
        
        aggregated = {
            'time_steps': time_steps,
            'n_seeds': self.n_seeds,
            'config': self.base_config,
            'by_time': {},
            'overall': {}
        }
        
        # Aggregate by time step
        for t in time_steps:
            coverage_rates = []
            interval_widths = []
            all_coverage_binary = []
            
            for seed, results in self.results_by_seed.items():
                if t in results:
                    coverage_rates.append(results[t]['coverage_rate'])
                    interval_widths.append(results[t]['interval_width'])
                    all_coverage_binary.extend(results[t]['coverage_history'])
            
            aggregated['by_time'][t] = {
                'coverage_mean': np.mean(coverage_rates),
                'coverage_std': np.std(coverage_rates),
                'coverage_median': np.median(coverage_rates),
                'coverage_q25': np.percentile(coverage_rates, 25),
                'coverage_q75': np.percentile(coverage_rates, 75),
                'coverage_min': np.min(coverage_rates),
                'coverage_max': np.max(coverage_rates),
                'width_mean': np.mean(interval_widths),
                'width_std': np.std(interval_widths),
                'width_median': np.median(interval_widths),
                'n_predictions_total': len(all_coverage_binary),
                'empirical_coverage': np.mean(all_coverage_binary),  # Pooled coverage
            }
        
        # Overall aggregated statistics (across all time steps and seeds)
        all_overall_coverage = []
        all_overall_widths = []
        
        for seed, results in self.results_by_seed.items():
            for t in time_steps:
                if t in results:
                    all_overall_coverage.extend(results[t]['coverage_history'])
                    n_pred = results[t]['n_predictions']
                    all_overall_widths.extend([results[t]['interval_width']] * n_pred)
        
        aggregated['overall'] = {
            'coverage_mean': np.mean(all_overall_coverage),
            'coverage_std': np.std(all_overall_coverage),
            'coverage_se': np.std(all_overall_coverage) / np.sqrt(len(all_overall_coverage)),
            'width_mean': np.mean(all_overall_widths),
            'width_std': np.std(all_overall_widths),
            'total_predictions': len(all_overall_coverage),
        }
        
        # Calculate time-based degradation statistics
        early_time_steps = time_steps[:len(time_steps)//3]
        late_time_steps = time_steps[-len(time_steps)//3:]
        
        early_coverages = [aggregated['by_time'][t]['coverage_mean'] for t in early_time_steps]
        late_coverages = [aggregated['by_time'][t]['coverage_mean'] for t in late_time_steps]
        
        aggregated['overall']['early_coverage_mean'] = np.mean(early_coverages)
        aggregated['overall']['late_coverage_mean'] = np.mean(late_coverages)
        aggregated['overall']['coverage_degradation'] = (
            np.mean(early_coverages) - np.mean(late_coverages)
        )
        
        return aggregated
    
    def plot_aggregated_results(self, aggregated: Dict, save_path: str = None):
        """
        Create comprehensive plots of aggregated results.
        
        Args:
            aggregated: Dictionary from aggregate_results()
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        time_steps = aggregated['time_steps']
        target_coverage = 1 - aggregated['config']['alpha']
        predictor_type = aggregated['config']['predictor']
        covariate_mode = aggregated['config']['covariate_mode']
        with_shift = aggregated['config']['with_shift']
        
        # Build title
        predictor_str = predictor_type.capitalize()
        if predictor_type == "algorithm":
            predictor_str = "Algorithm (AdaptedCAFHT)"
        mode_str = "Dynamic $X_t$" if covariate_mode == "dynamic" else "Static X"
        shift_str = "with Shift" if with_shift else "no Shift"
        main_title = f"Multi-Seed Aggregated Results (n={aggregated['n_seeds']}) — {predictor_str}, {mode_str}, {shift_str}"
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        
        # Plot 1: Coverage rate over time with confidence bands
        coverage_means = [aggregated['by_time'][t]['coverage_mean'] for t in time_steps]
        coverage_stds = [aggregated['by_time'][t]['coverage_std'] for t in time_steps]
        coverage_q25 = [aggregated['by_time'][t]['coverage_q25'] for t in time_steps]
        coverage_q75 = [aggregated['by_time'][t]['coverage_q75'] for t in time_steps]
        
        axes[0, 0].plot(time_steps, coverage_means, 'b-', linewidth=2, label='Mean Coverage')
        axes[0, 0].fill_between(time_steps, coverage_q25, coverage_q75, 
                                alpha=0.3, color='blue', label='IQR (25%-75%)')
        axes[0, 0].axhline(y=target_coverage, color='red', linestyle='--', 
                          linewidth=2, label=f'Target ({target_coverage:.1%})')
        axes[0, 0].set_ylim(0.8, 1.0)
        axes[0, 0].set_xlabel('Time Step t')
        axes[0, 0].set_ylabel('Coverage Rate')
        axes[0, 0].set_title(f'Coverage Rate vs. Time (Aggregated over {self.n_seeds} seeds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Interval width over time
        width_means = [aggregated['by_time'][t]['width_mean'] for t in time_steps]
        width_stds = [aggregated['by_time'][t]['width_std'] for t in time_steps]
        
        axes[0, 1].plot(time_steps, width_means, 'g-', linewidth=2)
        axes[0, 1].fill_between(time_steps, 
                                np.array(width_means) - np.array(width_stds),
                                np.array(width_means) + np.array(width_stds),
                                alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Time Step t')
        axes[0, 1].set_ylabel('Average Interval Width')
        axes[0, 1].set_title('Prediction Interval Width vs. Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of coverage rates across seeds at selected time points
        sample_times = [time_steps[0], time_steps[len(time_steps)//2], time_steps[-1]]
        sample_labels = ['Early', 'Middle', 'Late']
        
        coverage_distributions = []
        for t in sample_times:
            coverages_at_t = []
            for seed, results in self.results_by_seed.items():
                if t in results:
                    coverages_at_t.append(results[t]['coverage_rate'])
            coverage_distributions.append(coverages_at_t)
        
        bp = axes[0, 2].boxplot(coverage_distributions, labels=sample_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 2].axhline(y=target_coverage, color='red', linestyle='--', linewidth=2)
        axes[0, 2].set_ylabel('Coverage Rate')
        axes[0, 2].set_title('Coverage Distribution at Different Time Points')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Empirical coverage (pooled) vs nominal
        empirical_coverages = [aggregated['by_time'][t]['empirical_coverage'] for t in time_steps]
        
        axes[1, 0].plot(time_steps, empirical_coverages, 'b-', linewidth=2, label='Empirical (Pooled)')
        axes[1, 0].plot(time_steps, coverage_means, 'g--', linewidth=1.5, alpha=0.7, label='Mean across seeds')
        axes[1, 0].axhline(y=target_coverage, color='red', linestyle='--', linewidth=2, label='Target')
        axes[1, 0].set_xlabel('Time Step t')
        axes[1, 0].set_ylabel('Coverage Rate')
        axes[1, 0].set_title('Empirical vs. Mean Coverage')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0.8, 1.0)
        
        # Plot 5: Coverage variability across seeds
        axes[1, 1].plot(time_steps, coverage_stds, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time Step t')
        axes[1, 1].set_ylabel('Std Dev of Coverage')
        axes[1, 1].set_title('Coverage Variability Across Seeds')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics table
        axes[1, 2].axis('off')
        
        overall = aggregated['overall']
        table_data = [
            ['Overall Coverage', f"{overall['coverage_mean']:.3f} ± {overall['coverage_std']:.3f}"],
            ['Target Coverage', f"{target_coverage:.3f}"],
            ['Coverage Error', f"{overall['coverage_mean'] - target_coverage:+.3f}"],
            ['Early Coverage', f"{overall['early_coverage_mean']:.3f}"],
            ['Late Coverage', f"{overall['late_coverage_mean']:.3f}"],
            ['Degradation', f"{overall['coverage_degradation']:+.3f}"],
            ['Mean Width', f"{overall['width_mean']:.4f}"],
            ['Total Predictions', f"{overall['total_predictions']:,}"],
            ['Seeds', f"{self.n_seeds}"],
        ]
        
        table = axes[1, 2].table(cellText=table_data, cellLoc='left',
                                colWidths=[0.5, 0.5], loc='center',
                                bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#F5F5F5')
        
        axes[1, 2].set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_results(self, aggregated: Dict, filepath: str):
        """Save aggregated results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_data = convert_to_serializable(aggregated)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_summary(self, aggregated: Dict):
        """Print comprehensive summary of results."""
        print("\n" + "=" * 70)
        print("MULTI-SEED EXPERIMENT SUMMARY")
        print("=" * 70)
        
        config = aggregated['config']
        overall = aggregated['overall']
        target = 1 - config['alpha']
        
        print(f"\nConfiguration:")
        print(f"  Predictor       : {config['predictor']}")
        print(f"  Covariate mode  : {config['covariate_mode']}")
        print(f"  With shift      : {config['with_shift']}")
        print(f"  Seeds run       : {self.n_seeds}")
        print(f"  Series per seed : {config['n_series']}")
        print(f"  Time steps      : {config['T']}")
        
        print(f"\nOverall Results:")
        print(f"  Target coverage          : {target:.3f}")
        print(f"  Achieved coverage        : {overall['coverage_mean']:.3f} ± {overall['coverage_std']:.3f}")
        print(f"  Coverage error           : {overall['coverage_mean'] - target:+.3f}")
        print(f"  Standard error (pooled)  : {overall['coverage_se']:.4f}")
        print(f"  Total predictions made   : {overall['total_predictions']:,}")
        
        print(f"\nTime-Based Analysis:")
        print(f"  Early coverage (first 1/3) : {overall['early_coverage_mean']:.3f}")
        print(f"  Late coverage (last 1/3)   : {overall['late_coverage_mean']:.3f}")
        print(f"  Coverage degradation       : {overall['coverage_degradation']:+.3f}")
        
        print(f"\nInterval Width:")
        print(f"  Mean width      : {overall['width_mean']:.4f} ± {overall['width_std']:.4f}")
        
        # Statistical significance of coverage error
        z_score = (overall['coverage_mean'] - target) / overall['coverage_se']
        print(f"\nStatistical Assessment:")
        print(f"  Z-score for coverage error : {z_score:.2f}")
        print(f"  |Error| < 0.01 (1%)        : {'✓' if abs(overall['coverage_mean'] - target) < 0.01 else '✗'}")
        print(f"  |Error| < 0.02 (2%)        : {'✓' if abs(overall['coverage_mean'] - target) < 0.02 else '✗'}")
        print(f"  |Degradation| < 0.05 (5%)  : {'✓' if abs(overall['coverage_degradation']) < 0.05 else '✗'}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-seed conformal prediction experiments'
    )
    
    # Multi-seed specific arguments
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='Number of different seeds to run')
    parser.add_argument('--base_seed', type=int, default=1000,
                        help='Starting seed value')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Predictor and experiment configuration
    parser.add_argument('--predictor', choices=['basic', 'adaptive', 'algorithm'], 
                        default='algorithm',
                        help='Predictor type')
    parser.add_argument('--n_series', type=int, default=None,
                        help='Number of test series per seed')
    parser.add_argument('--n_train', type=int, default=None,
                        help='Number of training series')
    parser.add_argument('--n_cal', type=int, default=None,
                        help='Number of calibration series')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Miscoverage level')
    parser.add_argument('--aci_stepsize', type=float, default=0.005,
                        help='ACI stepsize for algorithm predictor')
    parser.add_argument('--T', type=int, default=None,
                        help='Time series length')
    
    # Adaptive-specific
    parser.add_argument('--window_size', type=int, default=800,
                        help='Window size for adaptive predictor')
    
    # Model parameters
    parser.add_argument('--ar_coef', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.2)
    parser.add_argument('--trend_coef', type=float, default=0.0)
    
    # Covariate configuration
    parser.add_argument('--covariate_mode', choices=['static', 'dynamic'], 
                        default='static')
    parser.add_argument('--with_shift', action='store_true')
    
    # Static covariate parameters
    parser.add_argument('--covar_rate', type=float, default=1.0)
    parser.add_argument('--covar_rate_shift', type=float, default=None)
    
    # Dynamic covariate parameters
    parser.add_argument('--x_rate', type=float, default=0.7)
    parser.add_argument('--x_trend', type=float, default=0.0)
    parser.add_argument('--x_noise_std', type=float, default=0.2)
    parser.add_argument('--x0_lambda', type=float, default=1.0)
    parser.add_argument('--x_rate_shift', type=float, default=None)
    parser.add_argument('--x_trend_shift', type=float, default=None)
    parser.add_argument('--x_noise_std_shift', type=float, default=None)
    parser.add_argument('--x0_lambda_shift', type=float, default=None)
    
    args = parser.parse_args()
    
    # Set predictor-specific defaults
    if args.predictor == "basic":
        defaults = {
            'n_series': 600,
            'n_train': 1200,
            'n_cal': 200,
            'T': 200,
            'covar_rate_shift': 3.0
        }
    elif args.predictor == "adaptive":
        defaults = {
            'n_series': 500,
            'n_train': 1000,
            'n_cal': 1000,
            'T': 40,
            'covar_rate_shift': 2.0
        }
    else:  # algorithm
        defaults = {
            'n_series': 500,
            'n_train': 1000,
            'n_cal': 1000,
            'T': 40,
            'covar_rate_shift': 2.0
        }
    
    # Apply defaults
    for key, default_val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_val)
    
    # Create configuration dictionary
    config = {
        'predictor': args.predictor,
        'n_series': args.n_series,
        'n_train': args.n_train,
        'n_cal': args.n_cal,
        'alpha': args.alpha,
        'aci_stepsize': args.aci_stepsize,
        'T': args.T,
        'window_size': args.window_size,
        'ar_coef': args.ar_coef,
        'beta': args.beta,
        'noise_std': args.noise_std,
        'trend_coef': args.trend_coef,
        'covariate_mode': args.covariate_mode,
        'with_shift': args.with_shift,
        'covar_rate': args.covar_rate,
        'covar_rate_shift': args.covar_rate_shift,
        'x_rate': args.x_rate,
        'x_trend': args.x_trend,
        'x_noise_std': args.x_noise_std,
        'x0_lambda': args.x0_lambda,
        'x_rate_shift': args.x_rate_shift,
        'x_trend_shift': args.x_trend_shift,
        'x_noise_std_shift': args.x_noise_std_shift,
        'x0_lambda_shift': args.x0_lambda_shift,
    }
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create experiment runner
    experiment = MultiSeedExperiment(
        base_config=config,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed
    )
    
    # Run all seeds
    start_time = datetime.now()
    results_by_seed = experiment.run_all_seeds()
    end_time = datetime.now()
    
    print(f"\nTotal runtime: {end_time - start_time}")
    
    # Aggregate results
    aggregated = experiment.aggregate_results()
    
    # Print summary
    experiment.print_summary(aggregated)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_str = config['predictor']
    shift_str = 'shift' if config['with_shift'] else 'noshift'
    
    # Save results
    results_file = save_dir / f"results_{pred_str}_{shift_str}_{timestamp}.json"
    experiment.save_results(aggregated, str(results_file))
    
    # Create and save plots
    plot_file = save_dir / f"plots_{pred_str}_{shift_str}_{timestamp}.png"
    experiment.plot_aggregated_results(aggregated, save_path=str(plot_file))
    
    print(f"\nAll results saved to {save_dir}/")


if __name__ == "__main__":
    main()
    