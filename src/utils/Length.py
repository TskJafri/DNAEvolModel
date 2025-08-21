#!/usr/bin/env python3
"""
Simple script to calculate the left-to-right size of a DNA structure in nanometers
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add paths for ipy_oxdna
sys.path.append('/home/tjafri/tsk')
sys.path.append('/home/tjafri/tsk/ipy_oxDNA')

try:
    # Check for availability but don't import globally
    import ipy_oxdna.dna_structure
    IPY_OXDNA_AVAILABLE = True
except ImportError:
    print("Warning: ipy_oxdna not available")
    IPY_OXDNA_AVAILABLE = False

def get_structure_length(top_file, dat_file):
    """Calculate the left-to-right length of a DNA structure in nm"""
    
    if not IPY_OXDNA_AVAILABLE:
        print("Error: ipy_oxdna is required for structure analysis")
        return None
    
    from ipy_oxdna.dna_structure import load_dna_structure
    
    try:
        # Load the DNA structure
        structure = load_dna_structure(top_file, dat_file)
        
        # Get all nucleotide positions
        all_positions = []
        for strand in structure.strands:
            for base in strand:
                all_positions.append(base.pos)
        
        positions = np.array(all_positions)
        
        # Find min and max in each dimension
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        
        # Calculate dimensions
        dimensions = max_coords - min_coords
        
        # Left-to-right is typically the X dimension (first coordinate)
        length_nm = dimensions[0]
        
        return {
            'length_x': dimensions[0],
            'length_y': dimensions[1], 
            'length_z': dimensions[2],
            'min_coords': min_coords,
            'max_coords': max_coords,
            'total_bases': len(all_positions)
        }
        
    except Exception as e:
        print(f"Error loading structure: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python3 Length.py <topology_file> <configuration_file>")
        print("Example: python3 Length.py structure.top structure.dat")
        return
    
    top_file = sys.argv[1]
    dat_file = sys.argv[2]
    
    # Check if files exist
    if not Path(top_file).exists():
        print(f"Error: Topology file '{top_file}' not found")
        return
    
    if not Path(dat_file).exists():
        print(f"Error: Configuration file '{dat_file}' not found")
        return
    
    print(f"Analyzing structure: {top_file}, {dat_file}")
    
    # Calculate structure dimensions
    result = get_structure_length(top_file, dat_file)
    
    if result:
        print(f"\nStructure Dimensions:")
        print(f"Left-to-right (X): {result['length_x']:.2f} nm")
        print(f"Width (Y):         {result['length_y']:.2f} nm") 
        print(f"Height (Z):        {result['length_z']:.2f} nm")
        print(f"Total nucleotides: {result['total_bases']}")
        print(f"\nCoordinate ranges:")
        print(f"X: {result['min_coords'][0]:.2f} to {result['max_coords'][0]:.2f}")
        print(f"Y: {result['min_coords'][1]:.2f} to {result['max_coords'][1]:.2f}")
        print(f"Z: {result['min_coords'][2]:.2f} to {result['max_coords'][2]:.2f}")
    else:
        print("Failed to analyze structure")

# Quick function for use in other scripts
def quick_length(top_file, dat_file):
    """Quick function to get just the left-to-right length"""
    result = get_structure_length(top_file, dat_file)
    return result['length_x'] if result else None

if __name__ == "__main__":
    main()
