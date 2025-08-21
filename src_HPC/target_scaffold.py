# src/target.py (Simplified Version)

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dna_model import DNAStructure

class TargetScaffold:
    def __init__(self, xyz_filepath: str):
        self.filepath: str = xyz_filepath
        self.raw_points: np.ndarray = self._load_xyz()
        self.path_length: float = self._calculate_path_length(self.raw_points)
        self.scaled_points: np.ndarray | None = None

        print(f"✅ Loaded scaffold from '{self.filepath}' with {len(self.raw_points)} points.")
        print(f"   - Scaffold path length: {self.path_length:.2f} units.")

    def _load_xyz(self) -> np.ndarray:
        points = []
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()[2:]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        points.append(list(map(float, parts[1:4])))
            return np.array(points)
        except Exception as e:
            print(f"❌ FATAL ERROR loading scaffold file: {e}"); raise

    def _calculate_path_length(self, points: np.ndarray) -> float:
        if len(points) < 2: return 0.0
        return np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))

    def scale_to_length(self, target_length: float):
        """Scales the raw scaffold points to a target path length."""
        if self.path_length == 0:
            self.scaled_points = self.raw_points.copy()
            return
            
        scale_factor = target_length / self.path_length
        origin = self.raw_points[0].copy()
        self.scaled_points = origin + (self.raw_points - origin) * scale_factor