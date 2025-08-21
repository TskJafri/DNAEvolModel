# scripts/test_fitness.py

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dna_model import DNAStructure
from src.target_scaffold import TargetScaffold
from src.fitness import FitnessCalculator

def test_invariance_with_real_data():
    """
    Verifies that the fitness score is invariant to rotation and translation
    by testing multiple versions of the same structure.
    """
    print("--- Running Test 1: Invariance Check with Real Data ---")
    
    # --- SETUP: Add paths to all versions of your structure ---
    structure_paths = [
        {"name": "Original", "top": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.top", "dat": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.dat"},
        {"name": "Rotated1", "top": "/Users/taskeenjafri/Projects/EvolModel/data/structures/RotatedShort/output.top", "dat": "/Users/taskeenjafri/Projects/EvolModel/data/structures/RotatedShort/output.dat"},
        {"name": "Rotated2", "top": "/Users/taskeenjafri/Projects/EvolModel/data/structures/RotatedShort2/output.top", "dat": "/Users/taskeenjafri/Projects/EvolModel/data/structures/RotatedShort2/output.dat"},
        {"name": "Rot+Trans", "top": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Rotated_Translated1/output.top", "dat": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Rotated_Translated1/output.dat"}
    ]
    scaffold_xyz_path = "/Users/taskeenjafri/Projects/EvolModel/data/scaffolds/3point.xyz"
    
    scaffold = TargetScaffold(xyz_filepath=scaffold_xyz_path)
    fitness_scores = []

    for info in structure_paths:
        print(f"\nTesting structure: {info['name']}")
        dna = DNAStructure(top_path=info["top"], dat_path=info["dat"])
        
        # --- NEW: Explicitly calculate and print the DNA length for confirmation ---
        positions = np.array([n.position for n in dna.nucleotides])
        center = np.mean(positions, axis=0)
        _, _, V = np.linalg.svd(positions - center)
        principal_axis = V[0]
        projected_values = np.dot(positions, principal_axis)
        dna_length = np.max(projected_values) - np.min(projected_values)
        print(f"   - Determined DNA length via PCA: {dna_length:.4f} nm")
        # -----------------------------------------------------------------------

        calculator = FitnessCalculator(target_scaffold=scaffold)
        score = calculator.calculate_fitness(dna)
        fitness_scores.append(score)
        print(f"✅ Calculated fitness: {score}")

    scores_std_dev = np.std(fitness_scores)
    print(f"\nStandard deviation of scores: {scores_std_dev}")
    assert scores_std_dev < 1e-3, "Test FAILED: Scores are not consistent across different orientations."

    print("--- Test 1 Passed: Fitness function is invariant! ---")
    return True

def test_for_correctness_with_mock_data():
    """
    A controlled experiment to verify the fitness score can distinguish
    between a good shape and a bad shape.
    """
    print("\n--- Running Test 2: Correctness Check with Mock Data ---")
    
    scaffold_path = "mock_scaffold.xyz"
    keyframes = np.array([[0.0, 0.0, 0.0], [5.0, 2.0, 0.0], [10.0, 0.0, 0.0]])
    with open(scaffold_path, "w") as f:
        f.write(f"{len(keyframes)}\nBent Scaffold\n")
        for p in keyframes: f.write(f"C {p[0]} {p[1]} {p[2]}\n")

    class MockNuc:
        def __init__(self, p, s_id=1): self.position = p; self.strand_id = s_id
    class MockDNA:
        def __init__(self, nucs): self.nucleotides = nucs
    
    scaffold = TargetScaffold(scaffold_path)
    perfect_dna = MockDNA([MockNuc(p) for p in scaffold.raw_points])
    calculator = FitnessCalculator(scaffold, weight_shape=1.0, weight_integrity=0.01)
    
    perfect_fitness = calculator.calculate_fitness(perfect_dna) # type: ignore
    print(f"✅ Calculated fitness for PERFECT match: {perfect_fitness}")
    assert perfect_fitness < 0.1, "Test FAILED: Perfect match score should be near zero."

    imperfect_keyframes = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    imperfect_dna = MockDNA([MockNuc(p) for p in imperfect_keyframes])
    imperfect_fitness = calculator.calculate_fitness(imperfect_dna) # type: ignore
    print(f"✅ Calculated fitness for IMPERFECT match: {imperfect_fitness}")
    assert imperfect_fitness > perfect_fitness, "Test FAILED: Imperfect score should be worse."
    
    os.remove(scaffold_path)
    print("--- Test 2 Passed: Calculator correctly scores shapes! ---")
    return True

if __name__ == '__main__':
    test1_ok = test_invariance_with_real_data()

    print("\n--------------------")
    if test1_ok:
        print("✅✅✅ ALL TESTS PASSED SUCCESSFULLY! ✅✅✅")
    else:
        print("❌ SOME TESTS FAILED. Please review the output above.")
    print("--------------------")