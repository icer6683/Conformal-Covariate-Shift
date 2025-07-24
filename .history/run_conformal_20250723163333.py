#!/usr/bin/env python3
"""
=============================================================================
COMPLETE CONFORMAL PREDICTION EXPERIMENT
=============================================================================

PURPOSE: End-to-end example showing how to use algorithm.py with ts_generator.py
         to validate coverage properties of the Adapted CAFHT algorithm.

This script demonstrates:
1. Generate time series data with covariate shift
2. Compute likelihood ratios for weighting  
3. Run Adapted CAFHT algorithm
4. Validate coverage properties (most important!)
5. Compare with/without covariate shift correction

USAGE:
  python run_conformal_experiment.py
  python run_conformal_experiment.py --n_test 100 --alpha 0.05
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List

# Import our modules
from fixed_ts_generator import TimeSeriesGenerator
from algorithm import AdaptedCAFHT, SimpleACI

def run_single_experiment(generator: TimeSeriesGenerator,
                         algorithm: AdaptedCAFHT,
                         train_data: np.ndarray,
                         test_data: np.ndarray,
                         likelihood_ratios: np.ndarray,
                         experiment_name: str) -> Dict:
    """
    Run the algorithm on a single test series and collect results.
    """
    n_test, T_plus_1, d = test_data.shape
    results = {
        'coverage_rates': [],
        'average_widths': [],
        'series_results': []
    }
    
    print(f"\n--- {experiment_name} ---")
    print(f"Processing {n_test} test series...")
    
    for i in range(n_test):
        # Split test series: observed vs true future
        test_series = test_data[i, :-1, :]  # Y_0 to Y_{T-1} (observed)
        true_future = test_data[i, :, :]    # Y_0 to Y_T (true values)
        
        # Get likelihood ratios for this series (use mean for simplicity)
        series_likelihood_ratios = likelihood_ratios[min(i, len(likelihood_ratios)-1)]
        
        # Run online algorithm
        try:
            prediction_bands, online_stats = algorithm.predict_online(
                D_cal=train_data,
                test_series=test_series,
                likelihood_ratios=np.full(train_data.shape[0]//2, series_likelihood_ratios),
                true_future=true_future
            )
            
            # Collect results
            coverage_rate = online_stats['final_coverage']
            avg_width = online_stats['average_width']
            
            results['coverage_rates'].append(coverage_rate)
            results['average_widths'].append(avg_width)
            results['series_results'].append(online_stats)
            
            if i < 5:  # Print first few for debugging
                print(f"  Series {i}: Coverage = {coverage_rate:.3f}, Width = {avg_width:.3f}")
                
        except Exception as e:
            print(f"  Error processing series {i}: {e}")
            continue
    
    # Aggregate results
    results['mean_coverage'] = np.mean(results['coverage_rates'])
    results['std_coverage'] = np.std(results['coverage_rates'])
    results['mean_width'] = np.mean(results['average_widths'])
    results['target_coverage'] = 1 - algorithm.alpha
    
    print(f"Results for {experiment_name}:")
    print(f"  Target coverage: {results['target_coverage']:.1%}")
    print(f"  Actual coverage: {results['mean_coverage']:.1%} ± {results['std_coverage']:.3f}")
    print(f"  Average width: {results['mean_width']:.4f}")
    print(f"  Coverage validity: {'✓' if abs(results['mean_coverage'] - results['target_coverage']) < 0.05 else '✗'}")
    
    return results

def compare_methods(train_data: np.ndarray,
                   original_test: np.ndarray, 
                   shifted_test: np.ndarray,
                   likelihood_ratios: np.ndarray,
                   alpha: float = 0.1) -> Dict:
    """
    Compare different conformal prediction approaches.
    """
    print("\n" + "="*80)
    print("CONFORMAL PREDICTION METHOD COMPARISON")
    print("="*80)
    
    results = {}
    
    # 1. Standard conformal (no covariate shift correction)
    print("\n1. STANDARD CONFORMAL PREDICTION (No shift correction)")
    standard_algorithm = AdaptedCAFHT(alpha=alpha)
    
    # Use uniform likelihood ratios (no weighting)
    uniform_ratios = np.ones(len(likelihood_ratios))
    
    results['standard_original'] = run_single_experiment(
        generator=None,
        algorithm=standard_algorithm,
        train_data=train_data,
        test_data=original_test,
        likelihood_ratios=uniform_ratios,
        experiment_name="Standard CP on Original Data"
    )
    
    results['standard_shifted'] = run_single_experiment(
        generator=None,
        algorithm=standard_algorithm, 
        train_data=train_data,
        test_data=shifted_test,
        likelihood_ratios=uniform_ratios,
        experiment_name="Standard CP on Shifted Data (SHOULD FAIL)"
    )
    
    # 2. Weighted conformal (with covariate shift correction)
    print("\n2. WEIGHTED CONFORMAL PREDICTION (With shift correction)")
    weighted_algorithm = AdaptedCAFHT(alpha=alpha)
    
    results['weighted_shifted'] = run_single_experiment(
        generator=None,
        algorithm=weighted_algorithm,
        train_data=train_data,
        test_data=shifted_test,
        likelihood_ratios=likelihood_ratios,
        experiment_name="Weighted CP on Shifted Data (SHOULD WORK)"
    )
    
    return results

def plot_coverage_comparison(results: Dict, save_plot: bool = False):
    """
    Create visualization comparing coverage across methods.
    """
    methods = ['Standard\n(Original)', 'Standard\n(Shifted)', 'Weighted\n(Shifted)']
    coverages = [
        results['standard_original']['mean_coverage'],
        results['standard_shifted']['mean_coverage'], 
        results['weighted_shifted']['mean_coverage']
    ]
    errors = [
        results['standard_original']['std_coverage'],
        results['standard_shifted']['std_coverage'],
        results['weighted_shifted']['std_coverage']
    ]
    target = results['standard_original']['target_coverage']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage plot
    bars = ax1.bar(methods, coverages, yerr=errors, capsize=5, alpha=0.7,
                   color=['blue', 'red', 'green'])
    ax1.axhline(y=target, color='black', linestyle='--', linewidth=2, 
                label=f'Target ({target:.1%})')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Coverage Comparison Across Methods')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add coverage validity indicators
    for i, (coverage, bar) in enumerate(zip(coverages, bars)):
        is_valid = abs(coverage - target) < 0.05
        color = 'green' if is_valid else 'red'
        symbol = '✓' if is_valid else '✗'
        ax1.text(bar.get_x() + bar.get_width()/2, coverage + 0.02, symbol,
                ha='center', va='bottom', fontsize=16, color=color, fontweight='bold')
    
    # Width comparison
    widths = [
        results['standard_original']['mean_width'],
        results['standard_shifted']['mean_width'],
        results['weighted_shifted']['mean_width']
    ]
    
    ax2.bar(methods, widths, alpha=0.7, color=['blue', 'red', 'green'])
    ax2.set_ylabel('Average Prediction Band Width')
    ax2.set_title('Efficiency Comparison (Lower = Better)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('conformal_prediction_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'conformal_prediction_comparison.png'")
    
    plt.show()

def main():
    """
    Complete experimental validation of conformal prediction coverage.
    """
    parser = argparse.ArgumentParser(description='Conformal Prediction Coverage Validation')
    parser.add_argument('--n_train', type=int, default=200, help='Training series')
    parser.add_argument('--n_test', type=int, default=30, help='Test series')
    parser.add_argument('--T', type=int, default=25, help='Time series length')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--shift_time', type=int, default=12, help='When covariate shift occurs')
    parser.add_argument('--shift_amount', type=float, default=2.0, help='Shift magnitude')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_plot', action='store_true', help='Save plots')
    
    args = parser.parse_args()
    
    print("CONFORMAL PREDICTION COVERAGE VALIDATION EXPERIMENT")
    print("="*60)
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (alpha = {args.alpha})")
    print(f"  Training series: {args.n_train}")
    print(f"  Test series: {args.n_test}")
    print(f"  Time series length: {args.T + 1}")
    print(f"  Covariate shift: {args.shift_amount} at t={args.shift_time}")
    
    # Generate data
    print(f"\nGenerating time series data...")
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    
    # Training data (no shift)
    train_data = generator.generate_ar_process(
        n=args.n_train,
        ar_coef=0.7,
        noise_std=0.2
    )
    
    # Test data with covariate shift
    original_test, shifted_test = generator.introduce_covariate_shift(
        train_data[:args.n_test],
        shift_time=args.shift_time,
        shift_params={'shift_amount': args.shift_amount}
    )
    
    # Compute likelihood ratios
    likelihood_ratios = generator.compute_likelihood_ratios(train_data, shifted_test)
    
    print(f"Data generated successfully!")
    print(f"  Training data: {train_data.shape}")
    print(f"  Test data: {shifted_test.shape}")
    print(f"  Likelihood ratios: mean = {np.mean(likelihood_ratios):.3f}")
    
    # Run comparison experiment
    results = compare_methods(
        train_data=train_data,
        original_test=original_test,
        shifted_test=shifted_test,
        likelihood_ratios=likelihood_ratios,
        alpha=args.alpha
    )
    
    # Create visualization
    plot_coverage_comparison(results, save_plot=args.save_plot)
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL COVERAGE VALIDATION SUMMARY")
    print("="*80)
    
    target_coverage = 1 - args.alpha
    
    print(f"\nTarget Coverage: {target_coverage:.1%}")
    print(f"\nMethod Performance:")
    
    methods_info = [
        ("Standard CP (Original Data)", results['standard_original'], "Should work well"),
        ("Standard CP (Shifted Data)", results['standard_shifted'], "Should FAIL due to covariate shift"),
        ("Weighted CP (Shifted Data)", results['weighted_shifted'], "Should work well with correction")
    ]
    
    for method_name, result, expectation in methods_info:
        coverage = result['mean_coverage']
        is_valid = abs(coverage - target_coverage) < 0.05
        status = "✓ PASS" if is_valid else "✗ FAIL"
        
        print(f"\n{method_name}:")
        print(f"  Actual coverage: {coverage:.1%}")
        print(f"  Status: {status}")
        print(f"  Expectation: {expectation}")
    
    # Key insight
    standard_shift_coverage = results['standard_shifted']['mean_coverage']
    weighted_shift_coverage = results['weighted_shifted']['mean_coverage']
    
    improvement = weighted_shift_coverage - standard_shift_coverage
    print(f"\nKey Result:")
    print(f"  Coverage improvement from weighting: {improvement:+.1%}")
    if improvement > 0.03:
        print(f"  ✓ Weighted conformal prediction successfully corrects for covariate shift!")
    else:
        print(f"  ⚠ Weighting may need adjustment or more data")
    
    print(f"\nExperiment completed!")

if __name__ == "__main__":
    main()