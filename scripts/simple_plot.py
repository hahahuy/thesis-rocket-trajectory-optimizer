#!/usr/bin/env python3
"""
Simple trajectory plotting script without pandas dependency
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

def read_csv_data(filename):
    """Read CSV data without pandas"""
    data = {}
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for key in reader.fieldnames:
                data[key] = []
            
            for row in reader:
                for key, value in row.items():
                    data[key].append(float(value))
                    
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        return data
    except FileNotFoundError:
        print(f"Error: CSV file '{filename}' not found.")
        print("Run './validate_dynamics' first to generate trajectory data.")
        return None

def plot_simple_trajectory(csv_file):
    """Create a simple trajectory plot"""
    
    data = read_csv_data(csv_file)
    if data is None:
        return
    
    # Use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Trajectory plot (x-y position)
    ax1.plot(data['x']/1000, data['y']/1000, 'b-', linewidth=2, label='Trajectory')
    ax1.plot(data['x'][0]/1000, data['y'][0]/1000, 'go', markersize=8, label='Start')
    ax1.plot(data['x'][-1]/1000, data['y'][-1]/1000, 'ro', markersize=8, label='End')
    ax1.set_xlabel('Horizontal Distance (km)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Rocket Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Altitude vs Time
    ax2.plot(data['time'], data['y']/1000, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Altitude vs Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity vs Time
    speed = np.sqrt(data['vx']**2 + data['vy']**2)
    ax3.plot(data['time'], data['vx'], 'r-', linewidth=2, label='Horizontal (vx)')
    ax3.plot(data['time'], data['vy'], 'b-', linewidth=2, label='Vertical (vy)')
    ax3.plot(data['time'], speed, 'k--', linewidth=2, label='Total speed')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Mass vs Time
    ax4.plot(data['time'], data['mass']/1000, 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Mass (tons)')
    ax4.set_title('Mass vs Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'rocket_trajectory_simple.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_file}")
    
    # Print summary
    print("\n=== Trajectory Summary ===")
    print(f"Flight time: {data['time'][-1]:.1f} seconds")
    print(f"Max altitude: {np.max(data['y'])/1000:.2f} km")
    print(f"Max speed: {np.max(speed):.1f} m/s ({np.max(speed)*3.6:.1f} km/h)")
    print(f"Horizontal range: {data['x'][-1]/1000:.2f} km")
    print(f"Final velocity: {speed[-1]:.1f} m/s")
    print(f"Mass consumed: {(data['mass'][0] - data['mass'][-1])/1000:.2f} tons")
    print(f"Final mass: {data['mass'][-1]/1000:.2f} tons")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'rocket_trajectory.csv'
    plot_simple_trajectory(csv_file)
