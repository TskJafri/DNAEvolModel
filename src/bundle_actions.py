import numpy as np
from sklearn.decomposition import PCA
from collections import deque
from typing import List, Optional

# Import the specific, user-provided classes
from .dna_model import DNAStructure, Nucleotide
from .oop_functions import truncate_strand


def nick_bundle(dna_structure: DNAStructure, percentage: float, out_top_path: str, out_dat_path: str) -> bool:
    """
    Creates a precise nick across the DNA bundle by slicing it with a
    mathematical plane, and saves the result.

    This function uses PCA to find the principal axis of the structure. It then
    nicks every single bond that crosses a plane defined at a specific `percentage`
    along that axis.

    Args:
        dna_structure (DNAStructure): The DNA structure object to modify.
        percentage (float): A value between 0.0 and 1.0 where the nick plane is.
        out_top_path (str): The output path for the new topology file.
        out_dat_path (str): The output path for the new coordinate file.

    Returns:
        bool: True if nicks were made and saved, False otherwise.
    """
    print(f"--- Starting Plane Nick at {percentage:.2%} ---")
    if not 0.0 < percentage < 1.0:
        print("❌ ERROR: Nick percentage must be between 0 and 1.")
        return False

    all_nucs = dna_structure.nucleotides
    if len(all_nucs) < 2:
        print("❌ ERROR: Not enough nucleotides for nicking.")
        return False

    # 1. Use PCA to find the structure's principal axis
    coords = np.array([nuc.position for nuc in all_nucs])
    pca = PCA(n_components=1)
    pca.fit(coords)
    principal_axis = pca.components_[0]
    center_point = pca.mean_

    # 2. Project all nucleotides onto the axis to get a 1D coordinate
    projections = np.dot(coords - center_point, principal_axis)
    nuc_to_proj = {nuc: proj for nuc, proj in zip(all_nucs, projections)}
    
    # 3. Determine the exact cut position in the 1D projection space
    min_proj, max_proj = np.min(projections), np.max(projections)
    cut_value = min_proj + percentage * (max_proj - min_proj)

    # 4. Identify every bond that crosses the plane using the corrected logic
    nicks_to_perform = []
    for n1 in all_nucs:
        n2 = n1.p3_neighbor
        if n2 is None:
            continue # No bond to nick

        proj1 = nuc_to_proj[n1]
        proj2 = nuc_to_proj[n2]

        # CORRECTED: Check if projections are on opposite sides of the cut_value.
        # This is the pure check for crossing the plane.
        if (proj1 < cut_value and proj2 > cut_value) or \
           (proj2 < cut_value and proj1 > cut_value):
            nicks_to_perform.append((n1, n2))

    if not nicks_to_perform:
        print("⚠️ No strands were found crossing the nicking plane. No action taken.")
        return False

    # 5. Execute all nicks (this logic remains the same)
    print(f"Found {len(nicks_to_perform)} bonds crossing the plane. Performing nicks...")
    max_strand_id = max(nuc.strand_id for nuc in all_nucs)
    
    for n1, n2 in nicks_to_perform:
        n1.p3_neighbor = None
        n2.p5_neighbor = None
        max_strand_id += 1
        new_strand_id = max_strand_id
        current = n2
        while current is not None:
            current.strand_id = new_strand_id
            current = current.p3_neighbor
            
    print(f"✅ In-memory nicks successful. Created {len(nicks_to_perform)} new strands.")
    
    # 6. Save the result
    dna_structure.save_files_minimal_change(out_top_path, out_dat_path)
    return True

def ligate_bundle(dna_structure: DNAStructure, out_top_path: str, out_dat_path: str) -> bool:
    """
    Intelligently ligates the two largest disconnected segments and saves the result.

    This function finds all connected components (segments), identifies the two largest,
    and uses a greedy algorithm to pair the closest compatible 3' and 5' ends between
    them. It performs the ligations and saves the result using the reordering method.

    Args:
        dna_structure (DNAStructure): The DNA structure object to modify.
        out_top_path (str): The output path for the new topology file.
        out_dat_path (str): The output path for the new coordinate file.
        
    Returns:
        bool: True if a ligation was performed and saved, False otherwise.
    """
    print("--- Starting Bundle Ligation ---")
    # 1. Find all connected components (segments) using Breadth-First Search (BFS)
    all_nucs_set = set(dna_structure.nucleotides)
    visited = set()
    segments = []
    
    while all_nucs_set:
        start_node = all_nucs_set.pop()
        if start_node in visited: continue

        component = []
        q = deque([start_node])
        visited.add(start_node)
        
        while q:
            current_nuc = q.popleft()
            component.append(current_nuc)
            # Explore neighbors using the correct attribute names
            for neighbor in [current_nuc.p5_neighbor, current_nuc.p3_neighbor]:
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        segments.append(component)
        all_nucs_set -= set(component)

    # 2. Select the two largest segments
    if len(segments) < 2:
        print("Fewer than two segments found. No ligation possible.")
        return False

    segments.sort(key=len, reverse=True)
    segment1, segment2 = segments[0], segments[1]
    print(f"Identified two largest segments with {len(segment1)} and {len(segment2)} nucleotides.")

    # 3. Find all loose 3' and 5' ends for each segment
    s1_ends_3 = [n for n in segment1 if n.p3_neighbor is None]
    s1_ends_5 = [n for n in segment1 if n.p5_neighbor is None]
    s2_ends_3 = [n for n in segment2 if n.p3_neighbor is None]
    s2_ends_5 = [n for n in segment2 if n.p5_neighbor is None]

    # 4. Find all possible ligation pairs and their distances
    potential_ligations = []
    # Pairs: 3' of seg1 -> 5' of seg2
    for n3 in s1_ends_3:
        for n5 in s2_ends_5:
            dist = np.linalg.norm(np.array(n3.position) - np.array(n5.position))
            potential_ligations.append((dist, n3, n5))
    # Pairs: 3' of seg2 -> 5' of seg1
    for n3 in s2_ends_3:
        for n5 in s1_ends_5:
            dist = np.linalg.norm(np.array(n3.position) - np.array(n5.position))
            potential_ligations.append((dist, n3, n5))
            
    if not potential_ligations:
        print("No compatible free ends found between the two largest segments.")
        return False

    # 5. Greedily perform ligations, starting with the closest pair
    potential_ligations.sort(key=lambda x: x[0])
    used_ends = set()
    ligations_performed = 0
    
    id1 = segment1[0].strand_id
    id2 = segment2[0].strand_id

    for dist, n3_end, n5_end in potential_ligations:
        if n3_end not in used_ends and n5_end not in used_ends:
            # Perform the ligation
            n3_end.p3_neighbor = n5_end
            n5_end.p5_neighbor = n3_end
            
            # Mark ends as used so they can't be ligated again
            used_ends.add(n3_end)
            used_ends.add(n5_end)
            ligations_performed += 1
            print(f"🧬 Ligating pair with distance: {dist:.3f} nm")

    if ligations_performed == 0:
        print("No ligations could be performed.")
        return False

    # 6. Merge the strand IDs, replicating the logic from target_ligate
    # We will merge the ID of the second segment into the first one.
    print(f"Merging strand {id2} into strand {id1}...")
    for nuc in segment2:
        nuc.strand_id = id1
    
    print(f"✅ In-memory ligation successful! Performed {ligations_performed} ligations.")
    
    # 7. Save the result using the method for reordering operations
    dna_structure.save_files_after_ligation(out_top_path, out_dat_path)
    return True

# In src/actions.py

def induce_bend(dna_structure: DNAStructure,
                percentage: float,
                direction_vector: list,
                severity: int = 1,
                num_strands_to_affect: int = 6, # New parameter for precise control
                dry_run: bool = False):
    """
    Induces a permanent bend by deleting nucleotides from the most extreme "inner" strands.

    This function identifies the strands that are furthest on the "inside" of a
    desired curve and applies a targeted deletion to them, creating precise strain.

    Args:
        dna_structure (DNAStructure): The structure to modify.
        percentage (float): The location of the bend (0.0 to 1.0).
        direction_vector (list): A 3D vector pointing "out" of the desired curve.
        severity (int): The number of NUCLEOTIDES to delete at the hinge.
        num_strands_to_affect (int): The number of extreme inner strands to modify.
        dry_run (bool): If True, prints a debug report instead of modifying the structure.
    
    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    print(f"--- Inducing Bend via Selective Deletion at {percentage:.2%} ---")
    if severity <= 0: return False

    all_nucs = dna_structure.nucleotides
    if not all_nucs: return False

    # 1. PCA and Hinge Point Calculation (same as before)
    coords = np.array([nuc.position for nuc in all_nucs])
    pca = PCA(n_components=1)
    pca.fit(coords)
    principal_axis = pca.components_[0]
    center_point = pca.mean_
    projections = np.dot(coords - center_point, principal_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    hinge_proj_value = min_proj + percentage * (max_proj - min_proj)
    hinge_center_3d = center_point + hinge_proj_value * principal_axis
    nuc_to_proj = {nuc: proj for nuc, proj in zip(all_nucs, projections)}

    # 2. NEW LOGIC: Calculate dot product for ALL strands and sort them
    strand_dot_products = []
    bend_dir = np.array(direction_vector) / np.linalg.norm(direction_vector)
    unique_strand_ids = sorted(list(set(nuc.strand_id for nuc in all_nucs)))
    
    for strand_id in unique_strand_ids:
        strand_nucs = [n for n in all_nucs if n.strand_id == strand_id]
        if not strand_nucs: continue
        closest_nuc = min(strand_nucs, key=lambda n: abs(nuc_to_proj[n] - hinge_proj_value))
        vec_to_nuc = np.array(closest_nuc.position) - hinge_center_3d
        dot_product = np.dot(vec_to_nuc, bend_dir)
        strand_dot_products.append({'id': strand_id, 'dot': dot_product})

    # Sort strands by dot product, from most negative (most "inside") to most positive
    strand_dot_products.sort(key=lambda x: x['dot'])

    # 3. Select only the top N most extreme "inside" strands
    # We only consider strands with a negative dot product to be eligible.
    eligible_inside_strands = [s for s in strand_dot_products if s['dot'] < 0]
    final_strands_to_modify = eligible_inside_strands[:num_strands_to_affect]
    inside_strand_ids = [s['id'] for s in final_strands_to_modify]

    # --- Dry Run Report ---
    if dry_run:
        print("\n--- INDUCE BEND (DRY RUN) ---")
        print(f"Using direction vector: {np.round(bend_dir, 2).tolist()}")
        print(f"Targeting the {len(inside_strand_ids)} most extreme 'INSIDE' strands.")
        print("-" * 45)
        for s in strand_dot_products:
            status = "TARGETED" if s['id'] in inside_strand_ids else "IGNORE"
            print(f"  - Strand {s['id']:<3}: Dot Product = {s['dot']:8.3f}  ->  {status}")
        print("--- END DRY RUN ---")
        return False

    # ... (The deletion logic from step 4 onwards remains exactly the same) ...
    if not inside_strand_ids:
        print("⚠️ No inside strands found for the given direction. No action taken.")
        return False

    nucs_to_delete = set()
    for strand_id in inside_strand_ids:
        strand_nucs = [n for n in all_nucs if n.strand_id == strand_id]
        for i in range(len(strand_nucs) - 1):
            n1 = strand_nucs[i]
            n2 = n1.p3_neighbor
            if n2 is None or n2 not in strand_nucs: continue
            proj1, proj2 = nuc_to_proj[n1], nuc_to_proj[n2]
            if (proj1 < hinge_proj_value and proj2 > hinge_proj_value) or \
               (proj2 < hinge_proj_value and proj1 > hinge_proj_value):
                current_nuc = n1
                start_of_deletion_chain = None
                for _ in range(severity):
                    start_of_deletion_chain = current_nuc
                    nucs_to_delete.add(current_nuc)
                    current_nuc = current_nuc.p5_neighbor
                    if current_nuc is None: break
                nuc_before_gap = start_of_deletion_chain.p5_neighbor if start_of_deletion_chain else None
                nuc_after_gap = n2
                if nuc_before_gap: nuc_before_gap.p3_neighbor = nuc_after_gap
                if nuc_after_gap: nuc_after_gap.p5_neighbor = nuc_before_gap
                break 

        if not inside_strand_ids:
            print("⚠️ No inside strands found for the given direction. No action taken.")
            return False

    nucs_to_delete = set()
    for strand_id in inside_strand_ids:
        strand_nucs = [n for n in all_nucs if n.strand_id == strand_id]
        
        # --- REVISED DELETION TARGETING LOGIC ---
        # Instead of finding a bond that CROSSES the plane, find the bond CLOSEST to it.
        
        possible_bonds = []
        for i in range(len(strand_nucs) - 1):
            n1 = strand_nucs[i]
            n2 = n1.p3_neighbor
            if n2 is None or n2 not in strand_nucs: continue

            proj1 = nuc_to_proj[n1]
            
            # Calculate this bond's distance from the hinge plane
            dist_to_hinge = abs(proj1 - hinge_proj_value)
            possible_bonds.append({'dist': dist_to_hinge, 'n1': n1, 'n2': n2})

        if not possible_bonds:
            continue # This strand has no valid bonds, skip to the next one.
            
        # Find the bond on this strand that is closest to our target location
        best_bond = min(possible_bonds, key=lambda x: x['dist'])
        
        # The nucleotide to start deleting from is the n1 of the closest bond
        start_del_nuc = best_bond['n1']
        
        # The nucleotide that will be on the other side of the new gap
        nuc_after_gap = best_bond['n2']
        
        # Perform the deletion starting from our newly found target
        current_nuc = start_del_nuc
        start_of_deletion_chain = None
        for _ in range(severity):
            start_of_deletion_chain = current_nuc
            if current_nuc:
                nucs_to_delete.add(current_nuc)
                current_nuc = current_nuc.p5_neighbor
            else:
                break # Reached the end of the strand

        nuc_before_gap = start_of_deletion_chain.p5_neighbor if start_of_deletion_chain else None

        # Re-wire the neighbors to create the gap
        if nuc_before_gap:
            nuc_before_gap.p3_neighbor = nuc_after_gap
        if nuc_after_gap:
            nuc_after_gap.p5_neighbor = nuc_before_gap
    
    if not nucs_to_delete:
        print("⚠️ Final check failed to identify any nucleotides for deletion.")
        return False
        
    dna_structure.nucleotides = [n for n in all_nucs if n not in nucs_to_delete]

    print(f"✅ Deleted a total of {len(nucs_to_delete)} nucleotides from inner strands.")
    return True