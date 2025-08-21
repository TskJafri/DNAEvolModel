# Contains the DNA model and file handling logic for DNA structures.
# Use in oop_functions.py for operations like ligation, nicking, and mutation.

from __future__ import annotations
from typing import Optional, List
import sys
from collections import defaultdict

class Nucleotide:
    """Represents a nucleotide, now including its physical coordinate data."""
    def __init__(self, original_index: int, strand_id: int, base_type: str):
        # Topological Info
        self.original_index: int = original_index
        self.strand_id: int = strand_id
        self.base_type: str = base_type
        self.p5_neighbor: Optional[Nucleotide] = None
        self.p3_neighbor: Optional[Nucleotide] = None
        
        # Physical Coordinate Info (from .dat file)
        self.position: List[float] = [0.0, 0.0, 0.0]
        self.bb_vector: List[float] = [0.0, 0.0, 0.0]  # Backbone vector
        self.n_vector: List[float] = [0.0, 0.0, 0.0]   # Normal vector

class DNAStructure:
    """Manages the entire DNA structure, loading and saving both topology and coordinate files."""
    def __init__(self, top_path: str, dat_path: str):
        self.nucleotides: List[Nucleotide] = []
        self.dat_header: List[str] = []
        self._load_files(top_path, dat_path)
    
    def _load_files(self, top_path: str, dat_path: str):
        """Loads and validates both the topology and coordinate files, merging them into one data model."""
        try:
            # Load Topology File
            with open(top_path, 'r') as f_top:
                top_header = f_top.readline().strip()
                if not top_header: raise ValueError("Topology file is empty.")
                total_top_nucs, _ = map(int, top_header.split())
                top_lines = [line.strip().split() for line in f_top if line.strip()]

            # Load Coordinate File
            with open(dat_path, 'r') as f_dat:
                self.dat_header = [next(f_dat) for _ in range(3)] # Store the 3-line header
                dat_lines = [line.strip().split() for line in f_dat if line.strip()]

            # Validation
            if len(top_lines) != total_top_nucs or len(dat_lines) != total_top_nucs:
                raise ValueError(f"Mismatch in nucleotide counts: top({len(top_lines)}), dat({len(dat_lines)}), header({total_top_nucs})")

            # Create Nucleotide objects with all data
            for i in range(total_top_nucs):
                top_parts = top_lines[i]
                dat_parts = dat_lines[i]
                
                nuc = Nucleotide(i, int(top_parts[0]), top_parts[1])
                nuc.position = [float(x) for x in dat_parts[0:3]]
                nuc.bb_vector = [float(x) for x in dat_parts[3:6]]
                nuc.n_vector = [float(x) for x in dat_parts[6:9]]
                self.nucleotides.append(nuc)

            # Second pass: Link neighbors
            for i, top_parts in enumerate(top_lines):
                p5_idx, p3_idx = int(top_parts[2]), int(top_parts[3])
                if p5_idx != -1: self.nucleotides[i].p5_neighbor = self.nucleotides[p5_idx]
                if p3_idx != -1: self.nucleotides[i].p3_neighbor = self.nucleotides[p3_idx]

        except Exception as e:
            print(f"❌ FATAL ERROR loading files: {e}", file=sys.stderr)
            raise

    def save_files_after_ligation(self, out_top_path: str, out_dat_path: str):
        """
        Saves the structure by physically reordering the nucleotides based on strand
        connectivity and writing new, synchronized topology and coordinate files.
        """
        print("Saving files with physical reordering of both topology and coordinates...")

        # Step 1: Group nucleotides by their current logical strand ID.
        strands = defaultdict(list)
        for nuc in self.nucleotides:
            strands[nuc.strand_id].append(nuc)

        # Step 2: Create the mapping from old IDs to new, sequential IDs.
        final_ids_map = {old_id: new_id for new_id, old_id in enumerate(sorted(strands.keys()), 1)}

        # Step 3: Build the final list of nucleotides in the new physical order.
        final_write_list = []
        for old_id in sorted(strands.keys()):
            new_id = final_ids_map[old_id]
            strand_nucleotides = strands[old_id]
            
            head = next((n for n in strand_nucleotides if n.p5_neighbor is None), None)
            if head is None and strand_nucleotides: head = strand_nucleotides[0] # Handle circular
            if not head: continue

            current_nuc, visited = head, set()
            while current_nuc and id(current_nuc) not in visited:
                visited.add(id(current_nuc))
                current_nuc.strand_id = new_id # Update the ID on the object
                final_write_list.append(current_nuc)
                current_nuc = current_nuc.p3_neighbor

        # Step 4: Write the new, reordered .dat file.
        with open(out_dat_path, 'w') as f_dat:
            f_dat.writelines(self.dat_header)
            for nuc in final_write_list:
                pos = ' '.join(map(str, nuc.position))
                bb = ' '.join(map(str, nuc.bb_vector))
                nv = ' '.join(map(str, nuc.n_vector))
                f_dat.write(f"{pos} {bb} {nv} 0 0 0 0 0 0\n")

        # Step 5: Write the new, synchronized topology file.
        obj_to_new_idx_map = {id(nuc): i for i, nuc in enumerate(final_write_list)}
        with open(out_top_path, 'w') as f_top:
            f_top.write(f"{len(final_write_list)} {len(final_ids_map)}\n")
            for nuc in final_write_list:
                p5 = obj_to_new_idx_map.get(id(nuc.p5_neighbor), -1)
                p3 = obj_to_new_idx_map.get(id(nuc.p3_neighbor), -1)
                f_top.write(f"{nuc.strand_id} {nuc.base_type} {p5} {p3}\n")
        
        print(f"✅ Successfully saved new topology to '{out_top_path}' and coordinates to '{out_dat_path}'.")

    def save_files_minimal_change(self, out_top_path: str, out_dat_path: str):
        """
        Saves the files for operations like nicking that preserve the physical order
        of nucleotides and their coordinates.
        """
        print("Saving files with minimal change (preserving physical order)...")
        
        # The final list IS the current nucleotide list. No reordering.
        final_write_list = self.nucleotides

        # Build the renumbering map based on the physical order of appearance.
        strand_id_map = {}
        next_new_strand_id = 1
        for nuc in final_write_list:
            if nuc.strand_id not in strand_id_map:
                strand_id_map[nuc.strand_id] = next_new_strand_id
                next_new_strand_id += 1
        
        # Create the index map from the unchanged nucleotide list order.
        obj_to_idx_map = {id(nuc): i for i, nuc in enumerate(final_write_list)}

        # Write the .dat file, which requires no changes as nothing was reordered.
        with open(out_dat_path, 'w') as f_dat:
            f_dat.writelines(self.dat_header)
            for nuc in final_write_list:
                pos = ' '.join(map(str, nuc.position))
                bb = ' '.join(map(str, nuc.bb_vector))
                nv = ' '.join(map(str, nuc.n_vector))
                f_dat.write(f"{pos} {bb} {nv} 0 0 0 0 0 0\n")

        # Write the topology file with updated connections and renumbered strands.
        with open(out_top_path, 'w') as f_top:
            f_top.write(f"{len(final_write_list)} {len(strand_id_map)}\n")
            for nuc in final_write_list:
                final_strand_id = strand_id_map[nuc.strand_id]
                p5 = obj_to_idx_map.get(id(nuc.p5_neighbor), -1)
                p3 = obj_to_idx_map.get(id(nuc.p3_neighbor), -1)
                f_top.write(f"{final_strand_id} {nuc.base_type} {p5} {p3}\n")
        
        print(f"✅ Successfully saved new topology to '{out_top_path}' and coordinates to '{out_dat_path}'.")
