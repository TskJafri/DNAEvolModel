# src/fitness.py (Simplified Version)

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dna_model import DNAStructure
    from src.target_scaffold import TargetScaffold

class FitnessCalculator:
    def __init__(self, target_scaffold: TargetScaffold, weight_shape: float = 1.0, weight_integrity: float = 1.0):
        self.target_scaffold = target_scaffold
        self.weight_shape = weight_shape
        self.weight_integrity = weight_integrity
        print("✅ FitnessCalculator initialized.")

    def _get_structure_properties(self, dna_structure: DNAStructure) -> tuple[np.ndarray | None, float]:
        positions = np.array([n.position for n in dna_structure.nucleotides])
        if positions.shape[0] < 2: return None, 0.0
        center = np.mean(positions, axis=0)
        _, _, V = np.linalg.svd(positions - center)
        principal_axis = V[0]
        
        projected_values = np.dot(positions, principal_axis)
        endpoint_idx = np.argmax(projected_values)
        endpoint_vector = positions[endpoint_idx] - center
        if np.dot(principal_axis, endpoint_vector) < 0:
            principal_axis *= -1
            
        projected_values = np.dot(positions, principal_axis)
        length = np.max(projected_values) - np.min(projected_values)
        return principal_axis, length

    def _extract_control_points(self, dna_structure: DNAStructure, principal_axis: np.ndarray, percentages: np.ndarray) -> np.ndarray:
        positions = np.array([n.position for n in dna_structure.nucleotides])
        projected_values = np.dot(positions, principal_axis)
        min_proj, max_proj = np.min(projected_values), np.max(projected_values)

        control_points = []
        for frac in percentages:
            target_proj_val = min_proj + frac * (max_proj - min_proj)
            sigma = (max_proj - min_proj) / (2 * len(percentages))
            if sigma < 1e-6: sigma = 1.0

            distances_sq = (projected_values - target_proj_val)**2
            weights = np.exp(-distances_sq / (2 * sigma**2))
            
            if np.sum(weights) < 1e-9:
                closest_idx = np.argmin(np.abs(projected_values - target_proj_val))
                control_points.append(positions[closest_idx])
            else:
                control_points.append(np.average(positions, axis=0, weights=weights))
        return np.array(control_points)

    def _calculate_aligned_rmsd(self, points_mob, points_ref) -> float:
        if points_mob.shape != points_ref.shape or points_mob.shape[0] == 0: return float('inf')
        centroid_mob = np.mean(points_mob, axis=0)
        centroid_ref = np.mean(points_ref, axis=0)
        mob_centered = points_mob - centroid_mob
        ref_centered = points_ref - centroid_ref
        rotation, rmsd = Rotation.align_vectors(mob_centered, ref_centered)
        return rmsd

    def calculate_fitness(self, dna_structure: DNAStructure) -> float:
        axis, length = self._get_structure_properties(dna_structure)
        if axis is None: return float('inf')

        self.target_scaffold.scale_to_length(length)
        scaled_target_points = self.target_scaffold.scaled_points
        if scaled_target_points is None: return float('inf')

        # Generate percentages from the scaffold's path length
        segment_lengths = np.sqrt(np.sum(np.diff(self.target_scaffold.raw_points, axis=0)**2, axis=1))
        path_positions = np.insert(np.cumsum(segment_lengths), 0, 0)
        percentages = path_positions / self.target_scaffold.path_length
        
        structure_points = self._extract_control_points(dna_structure, axis, percentages)
        if structure_points.shape[0] != len(scaled_target_points): return float('inf')

        f_shape = self._calculate_aligned_rmsd(scaled_target_points, structure_points)
        
        num_strands = len(set(n.strand_id for n in dna_structure.nucleotides))
        f_integrity = num_strands 
        
        total_fitness = (self.weight_shape * f_shape) + (self.weight_integrity * f_integrity)
        return total_fitness
    
    def find_worst_deviation(self, dna_structure: DNAStructure) -> tuple[float, np.ndarray] | None:
        """
        Analyzes the structure to find the point of maximum deviation from the target.

        Returns:
            A tuple containing the (percentage, direction_vector) for the worst spot,
            or None if the calculation fails.
        """
        axis, length = self._get_structure_properties(dna_structure)
        if axis is None: return None

        self.target_scaffold.scale_to_length(length)
        scaled_target_points = self.target_scaffold.scaled_points
        if scaled_target_points is None: return None

        segment_lengths = np.sqrt(np.sum(np.diff(self.target_scaffold.raw_points, axis=0)**2, axis=1))
        path_positions = np.insert(np.cumsum(segment_lengths), 0, 0)
        percentages = path_positions / self.target_scaffold.path_length
        
        structure_points = self._extract_control_points(dna_structure, axis, percentages)
        if structure_points.shape[0] != len(scaled_target_points): return None

        # Align the structures first to compare them in the same reference frame
        centroid_mob = np.mean(structure_points, axis=0)
        centroid_ref = np.mean(scaled_target_points, axis=0)
        mob_centered = structure_points - centroid_mob
        ref_centered = scaled_target_points - centroid_ref
        rotation, _ = Rotation.align_vectors(mob_centered, ref_centered)
        
        structure_points_aligned = rotation.apply(mob_centered) + centroid_ref
        
        # Now find the point with the largest squared distance error
        deviations = np.sum((structure_points_aligned - scaled_target_points)**2, axis=1)
        worst_point_index = np.argmax(deviations)
        
        # The corrective vector points from our structure's bad point TO the target point
        worst_structure_point = structure_points_aligned[worst_point_index]
        worst_target_point = scaled_target_points[worst_point_index]
        direction_vector = worst_target_point - worst_structure_point
        
        # The percentage is the location of this worst point
        percentage = percentages[worst_point_index]

        return percentage, direction_vector
