# scripts/test_target.py

import numpy as np
import sys
import os

# This adds the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dna_model import DNAStructure
from src.target_scaffold import TargetScaffold

def test_scaffold_scaling():
    """
    Tests that the TargetScaffold class correctly scales to the true
    length of a real, arbitrarily oriented DNA structure.
    """
    print("--- Running TargetScaffold Test ---")
    
    # --- SETUP: Define paths to your real data files ---
    structure_top_path = "/Users/taskeenjafri/Projects/EvolModel/data/structures/6Bundle/input.top"
    structure_dat_path = "/Users/taskeenjafri/Projects/EvolModel/data/structures/6Bundle/last_conf.dat"
    scaffold_xyz_path = "/Users/taskeenjafri/Projects/EvolModel/data/scaffolds/3point.xyz"

    # 1. Load the real DNA structure
    print("Loading DNA structure...")
    dna = DNAStructure(top_path=structure_top_path, dat_path=structure_dat_path)
    print(f"✅ Successfully loaded DNA structure with {len(dna.nucleotides)} nucleotides.")

    # 2. Calculate its true length using PCA (the "ground truth")
    positions = np.array([n.position for n in dna.nucleotides])
    center = np.mean(positions, axis=0)
    _, _, V = np.linalg.svd(positions - center)
    principal_axis = V[0]
    projected_values = np.dot(positions, principal_axis)
    dna_length = np.max(projected_values) - np.min(projected_values)
    print(f"   - Calculated true DNA length via PCA: {dna_length:.4f} nm")

    # 3. Instantiate the target scaffold and scale it to the calculated length
    print("\nLoading and scaling scaffold...")
    scaffold = TargetScaffold(xyz_filepath=scaffold_xyz_path)
    scaffold.scale_to_length(dna_length)
    
    # 4. VERIFY THE RESULTS
    print("\nVerifying results...")
    assert scaffold.scaled_points is not None, "Test FAILED: Scaled points should not be None."
    
    # Calculate the path length of the newly scaled scaffold
    scaled_length = scaffold._calculate_path_length(scaffold.scaled_points)
    print(f"   - Path length of scaled scaffold: {scaled_length:.4f} nm")
    
    # Assert that the scaled scaffold's length matches the DNA's true length
    assert abs(scaled_length - dna_length) < 1e-9, "Test FAILED: Scaled length does not match DNA length."
    
    print("\n✅✅✅ TargetScaffold test passed successfully! ✅✅✅")


if __name__ == "__main__":
    test_scaffold_scaling()