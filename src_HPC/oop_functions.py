# Uses DNA model to perform general, basic, PRECISE operations on DNA structures.
# These are not random, but deterministic operations that can be used in evolutionary models.
# The functions here are used in information/howtorunoopfunctions.py.

import random
from src.dna_model import DNAStructure, Nucleotide
import numpy as np
from sklearn.decomposition import PCA

def target_ligate(dna: DNAStructure, strand1_id: int, strand1_end: str, strand2_id: int, strand2_end: str) -> bool:
    """
    Performs a targeted ligation between specified ends of two different strands.
    This is a deterministic, precise operation.
    """
    # ... (code for target_ligate remains the same) ...
    print(f"Attempting to ligate {strand1_end} of strand {strand1_id} to {strand2_end} of strand {strand2_id}...")
    if strand1_id == strand2_id:
        print(f"❌ ERROR: Cannot ligate a strand to itself.")
        return False
    if strand1_end == strand2_end:
        print(f"❌ ERROR: Ligation must be between a 5' and a 3' end. Both were '{strand1_end}'.")
        return False
    if strand1_end == '3_prime':
        p3_strand_id, p5_strand_id = strand1_id, strand2_id
    else:
        p3_strand_id, p5_strand_id = strand2_id, strand1_id
    nuc_3_end = next((n for n in dna.nucleotides if n.strand_id == p3_strand_id and n.p3_neighbor is None), None)
    nuc_5_end = next((n for n in dna.nucleotides if n.strand_id == p5_strand_id and n.p5_neighbor is None), None)
    if not all([nuc_3_end, nuc_5_end]):
        print(f"❌ ERROR: Could not find a valid terminus for one or both strands.")
        if nuc_3_end is None: print(f"   - No 3' end found for strand {p3_strand_id}.")
        if nuc_5_end is None: print(f"   - No 5' end found for strand {p5_strand_id}.")
        return False
    assert nuc_3_end is not None
    assert nuc_5_end is not None
    nuc_3_end.p3_neighbor = nuc_5_end
    nuc_5_end.p5_neighbor = nuc_3_end
    for nuc in dna.nucleotides:
        if nuc.strand_id == p5_strand_id:
            nuc.strand_id = p3_strand_id
    print(f"🧬 In-memory ligation successful! Merged strand {p5_strand_id} into {p3_strand_id}.")
    print("   Ready to save the new, reordered structure.")
    return True

def target_nick(dna: DNAStructure, nick_after_idx: int) -> bool:
    """
    Performs a precise nick in the backbone after a specified nucleotide index.
    This is a minimal change operation that preserves physical file order.
    
    Args:
        dna: The DNAStructure object to modify.
        nick_after_idx: The 0-based index of the nucleotide to nick the bond AFTER.
        
    Returns:
        True if the nick was successful, False otherwise.
    """
    print(f"Attempting to nick the bond after nucleotide at index {nick_after_idx}...")
    
    # --- Validation ---
    if not (0 <= nick_after_idx < len(dna.nucleotides)):
        print(f"❌ ERROR: Index {nick_after_idx} is out of bounds.")
        return False
        
    nuc_to_nick = dna.nucleotides[nick_after_idx]
    
    if nuc_to_nick.p3_neighbor is None:
        print(f"❌ ERROR: Cannot nick at index {nick_after_idx}. It is already a 3' end.")
        return False

    three_prime_neighbor = nuc_to_nick.p3_neighbor
    assert three_prime_neighbor is not None

    # --- Perform the nick in memory ---
    nuc_to_nick.p3_neighbor = None
    three_prime_neighbor.p5_neighbor = None
    
    # Determine a new, unique temporary strand ID
    max_strand_id = max(nuc.strand_id for nuc in dna.nucleotides)
    new_strand_id = max_strand_id + 1
    
    # Traverse the new strand segment and update its ID
    current = three_prime_neighbor
    while current is not None:
        current.strand_id = new_strand_id
        current = current.p3_neighbor
    
    print(f"✅ In-memory nick successful. Created new temporary strand {new_strand_id}.")
    print("   Ready to save the structure with minimal changes.")
    return True

# def nick_bond(dna: DNAStructure):
#     """
#     Performs a nick. Does not change the save strategy, so the
#     default 'preserve_order' method will be used.
#     """
#     eligible_nucleotides = [nuc for nuc in dna.nucleotides if nuc.p3_neighbor is not None]
    
#     if not eligible_nucleotides:
#         print("⚠️ No bonds available to nick.")
#         return
    
#     nuc_to_nick = random.choice(eligible_nucleotides)
#     three_prime_neighbor = nuc_to_nick.p3_neighbor
    
#     # Pylance doesn't like that the p5 neighbor might be None, so we assert it exists.
#     assert three_prime_neighbor is not None

#     nuc_to_nick.p3_neighbor = None
#     three_prime_neighbor.p5_neighbor = None
    
#     max_strand_id = 0
#     if dna.nucleotides:
#         max_strand_id = max(nuc.strand_id for nuc in dna.nucleotides)
#     new_strand_id = max_strand_id + 1
    
#     current = three_prime_neighbor
#     while current is not None:
#         current.strand_id = new_strand_id
#         current = current.p3_neighbor
    
#     print(f"✅ Nicked bond after nucleotide with original index {nuc_to_nick.original_index}.")

def target_mutate(dna: DNAStructure, nuc_index: int, new_base: str) -> bool:
    """
    Mutates a specific nucleotide to a specific new base.
    This is a precise action for an evolutionary model.
    """
    all_bases = ['A', 'T', 'C', 'G']
    if not (0 <= nuc_index < len(dna.nucleotides)):
        print(f"❌ ERROR: Index {nuc_index} is out of bounds.")
        return False
    if new_base not in all_bases:
        print(f"❌ ERROR: Invalid base '{new_base}'.")
        return False
        
    chosen_nucleotide = dna.nucleotides[nuc_index]
    old_base = chosen_nucleotide.base_type
    chosen_nucleotide.base_type = new_base
    print(f"✅ Mutated nucleotide at index {nuc_index} from '{old_base}' to '{new_base}'.")
    return True

def random_mutate(dna: DNAStructure):
    """Randomly mutates one nucleotide. Good for generating variation."""
    if not dna.nucleotides: return
    chosen_nucleotide = random.choice(dna.nucleotides)
    current_base = chosen_nucleotide.base_type
    all_bases = ['A', 'T', 'C', 'G']
    other_bases = [base for base in all_bases if base != current_base]
    new_base = random.choice(other_bases)
    chosen_nucleotide.base_type = new_base
    print(f"✅ Randomly mutated nucleotide at index {chosen_nucleotide.original_index} to '{new_base}'.")
    return True

def remove_random_strand(dna: DNAStructure):
    # Step 1: Get a list of all unique strand_ids present in dna.nucleotides
    unique_strand_ids = list(set(nuc.strand_id for nuc in dna.nucleotides))
    
    # Step 2: If the list of IDs is empty, stop
    if not unique_strand_ids:
        return
    
    # Step 3: Randomly choose one strand_id from the list to be removed
    strand_id_to_remove = random.choice(unique_strand_ids)
    
    # Step 4: Create a list of all Nucleotide objects that are to be kept
    nucleotides_to_keep = [nuc for nuc in dna.nucleotides if nuc.strand_id != strand_id_to_remove]
    
    # Step 5: Check for connections to the removed strand and break them
    for nuc in nucleotides_to_keep:
        if nuc.p5_neighbor and nuc.p5_neighbor.strand_id == strand_id_to_remove:
            nuc.p5_neighbor = None
        if nuc.p3_neighbor and nuc.p3_neighbor.strand_id == strand_id_to_remove:
            nuc.p3_neighbor = None
    
    # Step 6: Replace the structure's nucleotide list with the list of nucleotides to keep
    dna.nucleotides = nucleotides_to_keep
    print(f"✅ Removed strand {strand_id_to_remove}")
    return True

def truncate_strand(dna: DNAStructure, strand_id: int, end_to_truncate: str, num_to_truncate: int) -> bool:
    """
    Truncates a specific number of nucleotides from a specific end of a specific strand.
    """
    unique_strand_ids = list(set(nuc.strand_id for nuc in dna.nucleotides))
    if strand_id not in unique_strand_ids:
        print(f"❌ ERROR: Strand {strand_id} not found.")
        return False
    if end_to_truncate not in ['5_prime', '3_prime']:
        print(f"❌ ERROR: End must be '5_prime' or '3_prime'.")
        return False
    if num_to_truncate <= 0:
        print(f"❌ ERROR: Number to truncate must be positive.")
        return False

    # Find the starting nucleotide for truncation
    start_node = None
    if end_to_truncate == '3_prime':
        start_node = next((n for n in dna.nucleotides if n.strand_id == strand_id and n.p3_neighbor is None), None)
    else: # 5_prime
        start_node = next((n for n in dna.nucleotides if n.strand_id == strand_id and n.p5_neighbor is None), None)

    if start_node is None:
        print(f"❌ ERROR: Could not find '{end_to_truncate}' end for strand {strand_id} (it may be circular).")
        return False

    # Collect nucleotides to remove by traversing from the chosen end
    nucleotides_to_remove = []
    current = start_node
    for _ in range(num_to_truncate):
        if current is None: break
        nucleotides_to_remove.append(current)
        current = current.p5_neighbor if end_to_truncate == '3_prime' else current.p3_neighbor
    
    # Identify the new end of the strand and sever the connection
    if nucleotides_to_remove:
        last_removed = nucleotides_to_remove[-1]
        if end_to_truncate == '3_prime' and last_removed.p5_neighbor:
            last_removed.p5_neighbor.p3_neighbor = None
        elif end_to_truncate == '5_prime' and last_removed.p3_neighbor:
            last_removed.p3_neighbor.p5_neighbor = None

    # Rebuild the main nucleotide list without the removed ones
    nucleotides_to_remove_set = set(nucleotides_to_remove)
    dna.nucleotides = [nuc for nuc in dna.nucleotides if nuc not in nucleotides_to_remove_set]
    
    print(f"✅ Truncated {len(nucleotides_to_remove)} nucleotides from the {end_to_truncate} end of strand {strand_id}.")
    return True


def random_deletion_in_zone(dna: DNAStructure, percentage: float, num_deletions: int = 1, zone_width: float = 0.10) -> bool:
    """
    Deletes a specified number of random nucleotides within a defined geometric zone.

    Args:
        dna (DNAStructure): The structure to modify.
        percentage (float): The center of the mutation zone (0.0 to 1.0).
        num_deletions (int): The number of nucleotides to delete.
        zone_width (float): The total width of the zone as a fraction of the
                            structure's length (e.g., 0.10 is +/- 5%).

    Returns:
        bool: True if deletions were performed, False otherwise.
    """
    print(f"--- Attempting {num_deletions} random deletion(s) in a {zone_width:.0%} wide zone at {percentage:.0%} ---")
    all_nucs = dna.nucleotides
    if not all_nucs: return False

    # 1. Define the mutation zone using PCA on the nucleotide positions
    coords = np.array([nuc.position for nuc in all_nucs])
    pca = PCA(n_components=1).fit(coords)
    principal_axis = pca.components_[0]
    center_point = pca.mean_
    projections = np.dot(coords - center_point, principal_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    length = max_proj - min_proj
    zone_center = min_proj + percentage * length
    
    half_width = (length * zone_width) / 2.0
    lower_bound = zone_center - half_width
    upper_bound = zone_center + half_width

    # 2. Find all eligible nucleotides inside the zone.
    # An eligible nucleotide must be internal (have both neighbors).
    candidates = [
        n for n in all_nucs 
        if lower_bound <= np.dot(np.array(n.position) - center_point, principal_axis) <= upper_bound 
        and n.p5_neighbor is not None and n.p3_neighbor is not None
    ]

    if len(candidates) < num_deletions:
        print(f"⚠️ Warning: Found only {len(candidates)} eligible nucleotides, but {num_deletions} were requested. No action taken.")
        return False

    # 3. Randomly select and delete the nucleotides
    nucs_to_delete = random.sample(candidates, num_deletions)
    for nuc in nucs_to_delete:
        nuc_before = nuc.p5_neighbor
        nuc_after = nuc.p3_neighbor
        
        # Re-wire using the correct attribute names from your codebase
        if nuc_before: nuc_before.p3_neighbor = nuc_after
        if nuc_after: nuc_after.p5_neighbor = nuc_before
    
    # 4. Update the main list in the DNAStructure object
    delete_set = set(nucs_to_delete)
    dna.nucleotides = [n for n in all_nucs if n not in delete_set]
    
    print(f"✅ Successfully deleted {len(nucs_to_delete)} nucleotides.")
    return True


def random_insertion_in_zone(dna: DNAStructure, percentage: float, num_insertions: int = 1, zone_width: float = 0.10) -> bool:
    """
    Inserts a specified number of random nucleotides within a defined geometric zone.

    Args:
        dna (DNAStructure): The structure to modify.
        percentage (float): The center of the mutation zone (0.0 to 1.0).
        num_insertions (int): The number of nucleotides to insert.
        zone_width (float): The total width of the zone as a fraction of the
                            structure's length.

    Returns:
        bool: True if insertions were performed, False otherwise.
    """
    print(f"--- Attempting {num_insertions} random insertion(s) in a {zone_width:.0%} wide zone at {percentage:.0%} ---")
    all_nucs = dna.nucleotides
    if not all_nucs: return False

    # 1. Define the mutation zone (same logic as deletion)
    coords = np.array([nuc.position for nuc in all_nucs])
    pca = PCA(n_components=1).fit(coords)
    principal_axis = pca.components_[0]
    center_point = pca.mean_
    projections = np.dot(coords - center_point, principal_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    length = max_proj - min_proj
    zone_center = min_proj + percentage * length
    half_width = (length * zone_width) / 2.0
    lower_bound = zone_center - half_width
    upper_bound = zone_center + half_width

    # 2. Find all eligible bonds (represented by the 5' nuc, n1) inside the zone
    candidates = [
        n1 for n1 in all_nucs 
        if n1.p3_neighbor is not None and
        lower_bound <= np.dot(np.array(n1.position) - center_point, principal_axis) <= upper_bound
    ]

    if len(candidates) < num_insertions:
        print(f"⚠️ Warning: Found only {len(candidates)} eligible bonds, but {num_insertions} were requested. No action taken.")
        return False

    # 3. Randomly select bonds, create new nucleotides, and insert them
    bonds_to_mutate = random.sample(candidates, num_insertions)
    for n1 in bonds_to_mutate:
        n2 = n1.p3_neighbor

        if n2 is None:
            continue

        # Create a new nucleotide using the exact Nucleotide class constructor
        new_nuc = Nucleotide(original_index=-1, strand_id=n1.strand_id, base_type=random.choice(['A', 'T', 'C', 'G']))
        
        # Place the new nucleotide physically between its future neighbors
        new_nuc.position = ((np.array(n1.position) + np.array(n2.position)) / 2.0).tolist()
        new_nuc.bb_vector = ((np.array(n1.bb_vector) + np.array(n2.bb_vector)) / 2.0).tolist()
        new_nuc.n_vector = ((np.array(n1.n_vector) + np.array(n2.n_vector)) / 2.0).tolist()
        
        # Re-wire the neighbors using the correct attribute names
        n1.p3_neighbor = new_nuc
        new_nuc.p5_neighbor = n1
        new_nuc.p3_neighbor = n2
        if n2: n2.p5_neighbor = new_nuc
        
        # Add the new object to the main list
        dna.nucleotides.append(new_nuc)

    print(f"✅ Successfully inserted {len(bonds_to_mutate)} nucleotides.")
    return True
