#!/usr/bin/env python3
"""
Visualize neural network training and validation loss from CSV file.

Usage:
    python plot_loss.py [loss_history.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_loss_history(csv_path='loss_history.csv'):
    """
    Plot training and validation loss from CSV file.

    Args:
        csv_path: Path to the loss_history.csv file
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / 'loss_plot.png'

    # Create figure with good size for readability
    plt.figure(figsize=(12, 6))

    # Plot training loss (all epochs)
    plt.plot(df['epoch'], df['train_loss'],
             label='Training Loss',
             color='blue',
             linewidth=1.5,
             alpha=0.8)

    # Plot validation loss (only where it exists)
    val_df = df.dropna(subset=['val_loss'])

    if not val_df.empty:
        plt.plot(val_df['epoch'], val_df['val_loss'],
                 label='Validation Loss',
                 color='red',
                 linewidth=2,
                 marker='o',
                 markersize=4,
                 alpha=0.8)

    # Formatting
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics
    final_train = df['train_loss'].iloc[-1]
    initial_train = df['train_loss'].iloc[0]
    improvement = ((initial_train - final_train) / initial_train) * 100

    stats_text = f'Initial Train Loss: {initial_train:.4f}\n'
    stats_text += f'Final Train Loss: {final_train:.4f}\n'
    stats_text += f'Improvement: {improvement:.1f}%'

    if not val_df.empty:
        final_val = val_df['val_loss'].iloc[-1]
        stats_text += f'\nFinal Val Loss: {final_val:.4f}'

        # Check for overfitting
        if final_val > final_train * 1.1:
            stats_text += '\n⚠ Possible overfitting detected'

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9,
             family='monospace')

    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Get CSV path from command line argument or use default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'loss_history.csv'

    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File '{csv_path}' not found!")
        sys.exit(1)

    # Generate plot
    print(f"Reading loss history from {csv_path}...")
    plot_loss_history(csv_path)
    print("Done!")
