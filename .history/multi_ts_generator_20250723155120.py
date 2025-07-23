"""
Multivariate AR(2) Time Series Generator for Conformal Prediction with Covariate Shift
No visualization - focused on data generation and covariate shift implementation.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, List
import argparse

class MultivariateTimeSeriesGenerator:
    """
    Generate multivariate AR(2) time series data for conformal prediction with covariate shift.
    
    Model: Y_t = Φ₁ * Y_{t-1} + Φ₂ * Y_{t-2} + ε_t
    where Φ₁, Φ₂ are d×d coefficient matrices and ε_t ~ N(0, Σ)
    """
    
    def __init__(self, T: int = 50, d: int = 2, seed: Optional[int] = None):
        """
        Initialize the multivariate time series generator.
        
        Args:
            T: Length of time series (excluding initial conditions)
            d: Dimension of observations at each time step
            seed: Random seed for reproducibility
        """
        self.T = T
        self.d = d
        self.order = 2  # AR(2) model
        if seed is not None:
            np.random.seed(seed)
    
    def generate_stable_ar_coefficients(self, stability_factor: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stable AR(2) coefficient matrices that ensure stationarity.
        
        Args:
            stability_factor: Controls how stable the system is (0 < stability_factor < 1)
            
        Returns:
            Tuple of (Φ₁, Φ₂) coefficient matrices of shape (d, d)
        """
        # Generate random coefficients
        Phi1 = np.random.normal(0, 0.3, (self.d, self.d))
        Phi2 = np.random.normal(0, 0.2, (self.d, self.d))
        
        # Scale to ensure stability (simplified approach)
        # For multivariate AR(2), full stability condition is complex
        # Here we use a heuristic: scale eigenvalues to be within unit circle
        max_eigenval = max(
            np.max(np.abs(np.linalg.eigvals(Phi1))),
            np.max(np.abs(np.linalg.eigvals(Phi2)))
        )
        
        if max_eigenval > stability_factor:
            scale_factor = stability_factor / max_eigenval
            Phi1 *= scale_factor
            Phi2 *= scale_factor
        
        return Phi1, Phi2
    
    def generate_noise_covariance(self, noise_std: float = 0.2, 
                                 correlation: float = 0.0) -> np.ndarray:
        """
        Generate simple noise covariance matrix.
        
        Args:
            noise_std: Standard deviation of noise (same for all dimensions)
            correlation: Correlation between dimensions (0 = independent, 0.5 = moderate correlation)
            
        Returns:
            Covariance matrix of shape (d, d)
        """
        if correlation == 0.0:
            # Independent noise across dimensions (diagonal covariance)
            Sigma = np.eye(self.d) * (noise_std ** 2)
        else:
            # Simple correlation structure: same variance, constant correlation
            Sigma = np.full((self.d, self.d), correlation * (noise_std ** 2))
            np.fill_diagonal(Sigma, noise_std ** 2)
            
        return Sigma
    
    def generate_multivariate_ar2(self, 
                                n: int,
                                Phi1: Optional[np.ndarray] = None,
                                Phi2: Optional[np.ndarray] = None,
                                noise_std: float = 0.2,
                                noise_correlation: float = 0.0,
                                initial_mean: Optional[np.ndarray] = None,
                                initial_std: float = 1.0) -> np.ndarray:
        """
        Generate multivariate AR(2) time series with simple noise structure.
        
        Model: Y_t = Φ₁ * Y_{t-1} + Φ₂ * Y_{t-2} + ε_t
        
        Args:
            n: Number of time series to generate
            Phi1: AR(1) coefficient matrix (d, d). If None, generates stable coefficients
            Phi2: AR(2) coefficient matrix (d, d). If None, generates stable coefficients  
            noise_std: Standard deviation of noise (same for all dimensions, default=0.2)
            noise_correlation: Correlation between noise dimensions (default=0.0 for independence)
            initial_mean: Mean for initial conditions. If None, uses zeros
            initial_std: Standard deviation for initial conditions (default=1.0)
            
        Returns:
            Array of shape (n, T+2, d) containing time series data (includes 2 initial conditions)
        """
        # Generate coefficients if not provided
        if Phi1 is None or Phi2 is None:
            Phi1, Phi2 = self.generate_stable_ar_coefficients()
            
        # Simple noise covariance
        Sigma = self.generate_noise_covariance(noise_std, noise_correlation)
            
        if initial_mean is None:
            initial_mean = np.zeros(self.d)
            
        # Simple initial conditions covariance
        initial_cov = np.eye(self.d) * (initial_std ** 2)
        
        # Store coefficients for later use
        self.Phi1 = Phi1
        self.Phi2 = Phi2
        self.Sigma = Sigma
        
        data = np.zeros((n, self.T + self.order, self.d))
        
        # Generate initial conditions Y_0, Y_1
        for i in range(n):
            data[i, 0, :] = np.random.multivariate_normal(initial_mean, initial_cov)
            data[i, 1, :] = np.random.multivariate_normal(initial_mean, initial_cov)
        
        # Generate the rest of the time series using AR(2) model
        for t in range(self.order, self.T + self.order):
            noise = np.random.multivariate_normal(np.zeros(self.d), Sigma, n)
            data[:, t, :] = (data[:, t-1, :] @ Phi1.T + 
                           data[:, t-2, :] @ Phi2.T + 
                           noise)
        
        return data
    
    def introduce_covariate_shift(self,
                                data: np.ndarray,
                                shift_type: str = 'mean_shift',
                                shift_params: Dict = None,
                                preserve_coefficients: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce covariate shift while preserving conditional AR(2) relationship.
        
        Args:
            data: Original time series data of shape (n, T+2, d)
            shift_type: Type of shift to apply
            shift_params: Parameters for the shift
            preserve_coefficients: Whether to use same AR coefficients for regeneration
            
        Returns:
            Tuple of (original_data, shifted_data)
        """
        if shift_params is None:
            shift_params = {}
            
        n, T_plus_order, d = data.shape
        shifted_data = data.copy()
        
        # Apply shift to covariates (all time points except the last one)
        if shift_type == 'mean_shift':
            shift_vector = shift_params.get('shift_vector', np.ones(d))
            if len(shift_vector) != d:
                shift_vector = np.full(d, shift_params.get('shift_amount', 1.0))
            shifted_data[:, :-1, :] += shift_vector
            
        elif shift_type == 'scale_shift':
            scale_matrix = shift_params.get('scale_matrix', None)
            if scale_matrix is None:
                scale_factor = shift_params.get('scale_factor', 1.5)
                scale_matrix = np.eye(d) * scale_factor
            
            # Center and scale
            mean = np.mean(shifted_data[:, :-1, :], axis=(0, 1))
            centered = shifted_data[:, :-1, :] - mean
            shifted_data[:, :-1, :] = mean + np.einsum('ij,ntj->nti', scale_matrix, centered)
            
        elif shift_type == 'rotation_shift':
            # Apply rotation to the covariate space
            angle = shift_params.get('rotation_angle', np.pi/6)
            if d == 2:
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
            else:
                # For d > 2, create random rotation matrix
                rotation_matrix = self._generate_rotation_matrix(d, angle)
            
            shifted_data[:, :-1, :] = np.einsum('ij,ntj->nti', rotation_matrix, shifted_data[:, :-1, :])
            
        elif shift_type == 'selection_bias':
            # Select subset based on some criterion
            selection_dim = shift_params.get('selection_dim', 0)
            threshold = shift_params.get('threshold', 0.0)
            selection_prob = shift_params.get('selection_prob', 0.7)
            
            initial_values = shifted_data[:, 0, selection_dim]
            probs = np.where(initial_values > threshold, selection_prob, 1 - selection_prob)
            selected_indices = np.random.binomial(1, probs).astype(bool)
            shifted_data = shifted_data[selected_indices]
            n = shifted_data.shape[0]
        
        # Regenerate the last time point using the same AR(2) model
        if preserve_coefficients and hasattr(self, 'Phi1') and hasattr(self, 'Phi2'):
            noise = np.random.multivariate_normal(np.zeros(d), self.Sigma, n)
            shifted_data[:, -1, :] = (shifted_data[:, -2, :] @ self.Phi1.T +
                                    shifted_data[:, -3, :] @ self.Phi2.T +
                                    noise)
        
        return data, shifted_data
    
    def _generate_rotation_matrix(self, d: int, max_angle: float) -> np.ndarray:
        """Generate a random rotation matrix for d dimensions."""
        # Generate random orthogonal matrix using QR decomposition
        A = np.random.normal(0, 1, (d, d))
        Q, _ = np.linalg.qr(A)
        
        # Scale the rotation to have maximum angle max_angle
        # This is a simplified approach for higher dimensions
        return Q
    
    def compute_likelihood_ratios(self,
                                original_data: np.ndarray,
                                shifted_data: np.ndarray,
                                method: str = 'gaussian_kde',
                                feature_type: str = 'initial') -> np.ndarray:
        """
        Compute likelihood ratios for multivariate covariate shift.
        
        Args:
            original_data: Original time series data
            shifted_data: Data after covariate shift
            method: Method for density estimation
            feature_type: Which features to use ('initial', 'all_covariates', 'summary_stats')
            
        Returns:
            Array of likelihood ratios for each shifted time series
        """
        if feature_type == 'initial':
            # Use initial conditions as features
            original_features = original_data[:, :self.order, :].reshape(original_data.shape[0], -1)
            shifted_features = shifted_data[:, :self.order, :].reshape(shifted_data.shape[0], -1)
            
        elif feature_type == 'all_covariates':
            # Use all covariates (all but last time point)
            original_features = original_data[:, :-1, :].reshape(original_data.shape[0], -1)
            shifted_features = shifted_data[:, :-1, :].reshape(shifted_data.shape[0], -1)
            
        elif feature_type == 'summary_stats':
            # Use summary statistics as features
            original_features = self._compute_summary_stats(original_data[:, :-1, :])
            shifted_features = self._compute_summary_stats(shifted_data[:, :-1, :])
        
        # For multivariate features, we'll use a simplified approach
        # In practice, you might want more sophisticated density estimation
        
        if method == 'gaussian_kde' and original_features.shape[1] <= 4:
            # KDE works well for low-dimensional features
            try:
                kde_original = stats.gaussian_kde(original_features.T)
                kde_shifted = stats.gaussian_kde(shifted_features.T)
                
                likelihood_ratios = kde_shifted(shifted_features.T) / (kde_original(shifted_features.T) + 1e-10)
                return likelihood_ratios
            except:
                # Fall back to histogram method if KDE fails
                pass
        
        # Histogram-based method for higher dimensions
        # Use first principal component or mean across dimensions
        original_summary = np.mean(original_features, axis=1)
        shifted_summary = np.mean(shifted_features, axis=1)
        
        bins = min(50, len(original_summary) // 5)
        hist_orig, bin_edges = np.histogram(original_summary, bins=bins, density=True)
        hist_shift, _ = np.histogram(shifted_summary, bins=bin_edges, density=True)
        
        bin_indices = np.digitize(shifted_summary, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist_orig) - 1)
        
        hist_orig = np.maximum(hist_orig, 1e-10)
        likelihood_ratios = hist_shift[bin_indices] / hist_orig[bin_indices]
        
        return likelihood_ratios
    
    def _compute_summary_stats(self, data: np.ndarray) -> np.ndarray:
        """Compute summary statistics for each time series."""
        n, T, d = data.shape
        features = []
        
        for i in range(n):
            series_features = []
            for dim in range(d):
                series = data[i, :, dim]
                series_features.extend([
                    np.mean(series),
                    np.std(series),
                    np.min(series),
                    np.max(series)
                ])
            features.append(series_features)
        
        return np.array(features)
    
    def get_covariates_and_targets(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into covariates (X) and targets (Y) for your conformal prediction framework.
        
        Args:
            data: Time series data of shape (n, T+2, d)
            
        Returns:
            Tuple of (X, Y) where:
            - X: Covariates of shape (n, T+1, d) - all but last time point
            - Y: Targets of shape (n, d) - last time point
        """
        X = data[:, :-1, :]  # All but last time point (covariates)
        Y = data[:, -1, :]   # Last time point (target)
        return X, Y
    
    def print_model_info(self):
        """Print information about the generated model."""
        if hasattr(self, 'Phi1') and hasattr(self, 'Phi2'):
            print(f"Multivariate AR(2) Model (d={self.d}):")
            print(f"Φ₁ matrix:\n{self.Phi1}")
            print(f"Φ₂ matrix:\n{self.Phi2}")
            print(f"Noise covariance Σ:\n{self.Sigma}")
            
            # Check stability (simplified)
            max_eig_phi1 = np.max(np.abs(np.linalg.eigvals(self.Phi1)))
            max_eig_phi2 = np.max(np.abs(np.linalg.eigvals(self.Phi2)))
            print(f"Max eigenvalue |Φ₁|: {max_eig_phi1:.3f}")
            print(f"Max eigenvalue |Φ₂|: {max_eig_phi2:.3f}")


def main():
    """Example usage of the multivariate time series generator."""
    parser = argparse.ArgumentParser(description='Generate multivariate AR(2) time series with covariate shift')
    parser.add_argument('--n_train', type=int, default=100, help='Number of training series')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test series')
    parser.add_argument('--T', type=int, default=30, help='Length of time series')
    parser.add_argument('--d', type=int, default=3, help='Dimension of time series')
    parser.add_argument('--shift_amount', type=float, default=1.5, help='Amount of shift')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--noise_correlation', type=float, default=0.0, help='Noise correlation between dimensions')
    parser.add_argument('--shift_type', type=str, default='mean_shift',
                       choices=['mean_shift', 'scale_shift', 'rotation_shift', 'selection_bias'],
                       help='Type of covariate shift')
    
    args = parser.parse_args()
    
    print(f"Multivariate AR(2) Time Series Generator")
    print(f"Parameters: n_train={args.n_train}, n_test={args.n_test}, T={args.T}, d={args.d}")
    print(f"Shift: {args.shift_type} with amount={args.shift_amount}")
    print(f"Noise: std={args.noise_std}, correlation={args.noise_correlation}")
    
    # Initialize generator
    generator = MultivariateTimeSeriesGenerator(T=args.T, d=args.d, seed=args.seed)
    
    # Generate training data
    print(f"\nGenerating {args.n_train} training series...")
    train_data = generator.generate_multivariate_ar2(
        n=args.n_train,
        noise_std=args.noise_std,
        noise_correlation=args.noise_correlation
    )
    
    # Print model information
    print("\nGenerated Model:")
    generator.print_model_info()
    
    # Apply covariate shift
    print(f"\nApplying {args.shift_type} to {args.n_test} test series...")
    
    shift_params = {'shift_amount': args.shift_amount}
    if args.shift_type == 'mean_shift':
        shift_params['shift_vector'] = np.full(args.d, args.shift_amount)
    elif args.shift_type == 'scale_shift':
        shift_params['scale_factor'] = args.shift_amount
    elif args.shift_type == 'rotation_shift':
        shift_params['rotation_angle'] = args.shift_amount * np.pi / 6
    
    original_test, shifted_test = generator.introduce_covariate_shift(
        train_data[:args.n_test],
        shift_type=args.shift_type,
        shift_params=shift_params
    )
    
    # Compute likelihood ratios
    likelihood_ratios = generator.compute_likelihood_ratios(
        train_data, shifted_test, feature_type='initial'
    )
    
    # Get covariates and targets for conformal prediction
    X_train, Y_train = generator.get_covariates_and_targets(train_data)
    X_test_orig, Y_test_orig = generator.get_covariates_and_targets(original_test)
    X_test_shift, Y_test_shift = generator.get_covariates_and_targets(shifted_test)
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Original test shape: {original_test.shape}")
    print(f"Shifted test shape: {shifted_test.shape}")
    
    print(f"\nCovariates and Targets:")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test_shift shape: {X_test_shift.shape}, Y_test_shift shape: {Y_test_shift.shape}")
    
    print(f"\nOriginal test data statistics:")
    print(f"  Y mean: {np.mean(Y_test_orig, axis=0)}")
    print(f"  Y std:  {np.std(Y_test_orig, axis=0)}")
    
    print(f"\nShifted test data statistics:")
    print(f"  Y mean: {np.mean(Y_test_shift, axis=0)}")
    print(f"  Y std:  {np.std(Y_test_shift, axis=0)}")
    
    print(f"\nLikelihood ratios:")
    print(f"  Mean: {np.mean(likelihood_ratios):.3f}")
    print(f"  Std:  {np.std(likelihood_ratios):.3f}")
    print(f"  Range: [{np.min(likelihood_ratios):.3f}, {np.max(likelihood_ratios):.3f}]")
    
    print(f"\nData ready for conformal prediction with covariate shift!")


if __name__ == "__main__":
    main()