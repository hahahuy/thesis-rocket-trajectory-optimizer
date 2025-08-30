#!/usr/bin/env python3
"""
Plot rocket trajectory data from CSV output
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_trajectory(csv_file, output_dir=None):
    """Plot trajectory data from CSV file"""
    
    # Read the data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("Run './validate_dynamics' first to generate trajectory data.")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Trajectory plot (x-y position)
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(df['x']/1000, df['y']/1000, 'b-', linewidth=2, label='Trajectory')
    plt.plot(df['x'].iloc[0]/1000, df['y'].iloc[0]/1000, 'go', markersize=8, label='Start')
    plt.plot(df['x'].iloc[-1]/1000, df['y'].iloc[-1]/1000, 'ro', markersize=8, label='End')
    plt.xlabel('Horizontal Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.title('Rocket Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    # 2. Altitude vs Time
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(df['time'], df['altitude_km'], 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.title('Altitude vs Time')
    plt.grid(True, alpha=0.3)
    
    # 3. Velocity components vs Time
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(df['time'], df['vx'], 'r-', linewidth=2, label='Horizontal (vx)')
    plt.plot(df['time'], df['vy'], 'b-', linewidth=2, label='Vertical (vy)')
    plt.plot(df['time'], df['speed_ms'], 'k--', linewidth=2, label='Total speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Mass vs Time
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(df['time'], df['mass']/1000, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (tons)')
    plt.title('Mass vs Time')
    plt.grid(True, alpha=0.3)
    
    # 5. Energy vs Time
    ax5 = plt.subplot(2, 3, 5)
    # Calculate kinetic and potential energy separately
    kinetic_energy = 0.5 * df['mass'] * df['speed_ms']**2 / 1e6  # MJ
    potential_energy = df['mass'] * 9.81 * df['y'] / 1e6  # MJ
    total_energy = kinetic_energy + potential_energy
    
    plt.plot(df['time'], kinetic_energy, 'r-', linewidth=2, label='Kinetic')
    plt.plot(df['time'], potential_energy, 'b-', linewidth=2, label='Potential') 
    plt.plot(df['time'], total_energy, 'k-', linewidth=2, label='Total')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (MJ)')
    plt.title('Energy vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Flight path angle and acceleration
    ax6 = plt.subplot(2, 3, 6)
    # Calculate flight path angle
    flight_path_angle = np.arctan2(df['vy'], df['vx']) * 180 / np.pi
    
    plt.plot(df['time'], flight_path_angle, 'purple', linewidth=2, label='Flight path angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Path Angle (degrees)')
    plt.title('Flight Path Angle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        output_file = os.path.join(output_dir, 'rocket_trajectory.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n=== Trajectory Summary ===")
    print(f"Flight time: {df['time'].iloc[-1]:.1f} seconds")
    print(f"Max altitude: {df['altitude_km'].max():.2f} km")
    print(f"Max speed: {df['speed_ms'].max():.1f} m/s ({df['speed_ms'].max()*3.6:.1f} km/h)")
    print(f"Horizontal range: {df['x'].iloc[-1]/1000:.2f} km")
    print(f"Final velocity: {df['speed_ms'].iloc[-1]:.1f} m/s")
    print(f"Mass consumed: {(df['mass'].iloc[0] - df['mass'].iloc[-1])/1000:.2f} tons")
    print(f"Final mass: {df['mass'].iloc[-1]/1000:.2f} tons")
    
    # Calculate some performance metrics
    max_alt_idx = df['altitude_km'].idxmax()
    time_to_max_alt = df['time'].iloc[max_alt_idx]
    print(f"Time to max altitude: {time_to_max_alt:.1f} seconds")
    
    # Average vertical acceleration in first 10 seconds
    early_phase = df[df['time'] <= 10.0]
    if len(early_phase) > 1:
        avg_accel = np.mean(np.diff(early_phase['vy']) / np.diff(early_phase['time']))
        print(f"Average vertical acceleration (0-10s): {avg_accel:.2f} m/s²")

def main():
    parser = argparse.ArgumentParser(description='Plot rocket trajectory from CSV data')
    parser.add_argument('csv_file', nargs='?', default='rocket_trajectory.csv',
                        help='Path to trajectory CSV file (default: rocket_trajectory.csv)')
    parser.add_argument('-o', '--output', help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show plots interactively')
    
    args = parser.parse_args()
    
    # Check if running in non-interactive environment
    if args.no_show or not os.environ.get('DISPLAY'):
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        if not args.output:
            args.output = '.'  # Save to current directory
    
    plot_trajectory(args.csv_file, args.output if args.output else None)

if __name__ == '__main__':
    main()
