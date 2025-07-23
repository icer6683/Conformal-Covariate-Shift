#!/usr/bin/env python3
"""
Simple test version to debug syntax issues
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Simple test for time series generation')
    parser.add_argument('--T', type=int, default=30, help='Time series length')
    parser.add_argument('--shift_time', type=int, default=0, help='When shift occurs')
    parser.add_argument('--shift_amount', type=float, default=2.0, help='Amount of shift')
    
    args = parser.parse_args()
    
    print(f"Test parameters:")
    print(f"  T = {args.T}")
    print(f"  shift_time = {args.shift_time}")
    print(f"  shift_amount = {args.shift_amount}")
    
    # Simple validation
    if args.shift_time >= args.T:
        print(f"Error: shift_time ({args.shift_time}) must be < T ({args.T})")
        return
    
    # Generate simple test data
    np.random.seed(42)
    n = 10
    data = np.random.normal(0, 1, (n, args.T + 1, 1))
    
    # Apply simple shift
    shifted_data = data.copy()
    if args.shift_time == 0:
        shifted_data[:, :-1, :] += args.shift_amount
    else:
        shifted_data[:, args.shift_time:-1, :] += args.shift_amount
    
    print("Test completed successfully!")
    print(f"Original data shape: {data.shape}")
    print(f"Shifted data shape: {shifted_data.shape}")
    
    # Simple plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for i in range(min(3, n)):
        plt.plot(data[i, :, 0], 'b-', alpha=0.7)
    plt.title('Original')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i in range(min(3, n)):
        if args.shift_time > 0:
            plt.plot(range(args.shift_time), shifted_data[i, :args.shift_time, 0], 'b-', alpha=0.7)
            plt.plot(range(args.shift_time, args.T + 1), shifted_data[i, args.shift_time:, 0], 'r-', alpha=0.7)
            plt.axvline(x=args.shift_time, color='black', linestyle='--')
        else:
            plt.plot(shifted_data[i, :, 0], 'r-', alpha=0.7)
    plt.title(f'Shifted (t={args.shift_time})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()