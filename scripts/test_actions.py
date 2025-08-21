# scripts/test_actions.py

import os
import sys
# Make sure to import from 'src.actions', which is the correct file name
from src.dna_model import DNAStructure
from src.bundle_actions import nick_bundle, ligate_bundle, induce_bend
from src.oop_functions import random_deletion_in_zone, random_insertion_in_zone

# --- Test Functions ---

def test_nicking():
    """
    Tests the nick_bundle function on an intact structure.
    """
    print("--- Testing nick_bundle ---")
    
    # 1. Define input and output files
    input_top = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.top"
    input_dat = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.dat"
    
    output_dir = "/Users/taskeenjafri/Projects/EvolModel/output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_top = os.path.join(output_dir, "nicked_structure.top")
    output_dat = os.path.join(output_dir, "nicked_structure.dat")

    # 2. Load the structure
    print(f"Loading structure from '{input_top}'...")
    try:
        dna = DNAStructure(top_path=input_top, dat_path=input_dat)
    except Exception as e:
        print(f"❌ Failed to load DNA structure. Error: {e}")
        return

    # 3. Run the nicking function
    nick_bundle(dna, percentage=0.5, out_top_path=output_top, out_dat_path=output_dat)

def test_nick_then_ligate():
    """
    Tests the full cycle: takes an INTACT structure, nicks it, and then ligates it.
    This is a self-contained test of both functions working together.
    """
    print("--- Testing Full Cycle: Nick -> Ligate ---")

    # 1. Define file paths. CRITICAL: Start with an original, intact file.
    output_dir = "/Users/taskeenjafri/Projects/EvolModel/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start with the ORIGINAL structure, NOT a nicked one.
    initial_top = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.top"
    initial_dat = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.dat"

    # We'll save the intermediate nicked file here
    nicked_top = os.path.join(output_dir, "temp_nicked.top")
    nicked_dat = os.path.join(output_dir, "temp_nicked.dat")

    # This will be our final, repaired output
    final_ligated_top = os.path.join(output_dir, "cycle_test_ligated.top")
    final_ligated_dat = os.path.join(output_dir, "cycle_test_ligated.dat")

    # --- STEP 1: Nick the intact structure ---
    print("\n[Step 1/2] Nicking the original structure...")
    try:
        initial_dna = DNAStructure(top_path=initial_top, dat_path=initial_dat)
        success = nick_bundle(initial_dna, percentage=0.5, out_top_path=nicked_top, out_dat_path=nicked_dat)
        if not success:
            print("❌ Nicking step failed to produce an output. Aborting.")
            return
    except Exception as e:
        print(f"❌ Failed during the nicking step. Error: {e}")
        return
    print("[Step 1/2] Successfully created intermediate nicked structure.")

    # --- STEP 2: Ligate the fragmented structure ---
    print("\n[Step 2/2] Attempting to ligate the new fragmented structure...")
    try:
        nicked_dna = DNAStructure(top_path=nicked_top, dat_path=nicked_dat)
        ligate_bundle(nicked_dna, out_top_path=final_ligated_top, out_dat_path=final_ligated_dat)
    except Exception as e:
        print(f"❌ Failed during the ligation step. Error: {e}")
        return
    print("[Step 2/2] Ligation process complete.")

def test_ligate_only():
    """
    Tests ONLY the ligate_bundle function on a PRE-EXISTING nicked file.
    """
    print("--- Testing Ligate Only ---")

    # 1. Define file paths. The input here IS an already-nicked file.
    output_dir = "/Users/taskeenjafri/Projects/EvolModel/output"
    os.makedirs(output_dir, exist_ok=True)

    # Use your existing nicked file as the input for this test
    nicked_input_top = "/Users/taskeenjafri/Projects/EvolModel/output/nicked_bundle2.top"
    nicked_input_dat = "/Users/taskeenjafri/Projects/EvolModel/output/nicked_bundle2.dat"

    # Define the final output path
    final_ligated_top = os.path.join(output_dir, "direct_ligation_test.top")
    final_ligated_dat = os.path.join(output_dir, "direct_ligation_test.dat")
    
    # 2. Load the pre-nicked structure and ligate it
    print(f"\nLoading pre-nicked structure from '{nicked_input_top}'...")
    try:
        nicked_dna = DNAStructure(top_path=nicked_input_top, dat_path=nicked_input_dat)
        ligate_bundle(nicked_dna, out_top_path=final_ligated_top, out_dat_path=final_ligated_dat)
    except Exception as e:
        print(f"❌ Failed during the ligation step. Error: {e}")
        return
    print("Ligation process complete.")

def test_induce_bend():
    """
    Tests the induce_bend function to create a strained structure.
    """
    print("--- Testing Induce Bend ---")
    output_dir = "/Users/taskeenjafri/Projects/EvolModel/output/6Bundle"
    os.makedirs(output_dir, exist_ok=True)
    input_top = "/Users/taskeenjafri/Projects/EvolModel/output/6Bundle/bent_structure.top"
    input_dat = "/Users/taskeenjafri/Projects/EvolModel/output/6Bundle/last_conf.dat"
    output_top = os.path.join(output_dir, "bent_structure.top")
    output_dat = os.path.join(output_dir, "last_conf.dat")

    # --- SET BEND PARAMETERS ---
    bend_percentage = 0.1
    bend_severity = 3
    
    # Try different vectors to find the correct orientation for your bundle.
    # The goal is to find a vector that selects only 3-4 strands on one side.
    bend_direction = [0, 1, 0]  # Try this first
    # bend_direction = [0, -1, 0] # Then try this
    # bend_direction = [1, 0, 0]  # Then try this, etc.
    # bend_direction = [0, 0, 1]

    print(f"Loading intact structure from '{input_top}'...")
    try:
        dna = DNAStructure(top_path=input_top, dat_path=input_dat)
    except Exception as e:
        print(f"❌ Failed to load DNA structure. Error: {e}")
        return

    # --- RUN THE TEST ---
    # Set dry_run=True to see the report without changing the file.
    # Set dry_run=False to perform the actual truncation and save the file.
    success = induce_bend(dna_structure=dna,
                          percentage=bend_percentage,
                          direction_vector=bend_direction,
                          severity=bend_severity,
                          dry_run=False) # <-- Use the dry run mode!
    

    if success:
        print("\nBend induced in memory. Now saving the new structure...")
        dna.save_files_after_ligation(output_top, output_dat)
    else:
        print("\nDry run complete or action failed. No file was saved.")

def test_zone_mutations():
    """
    Tests the random deletion and insertion functions.
    """
    print("--- Testing Zonal Mutations ---")
    
    # --- Configuration ---
    # NOTE: Please update these paths to a valid input file for your system
    output_dir = "/Users/taskeenjafri/Projects/EvolModel/output/Short"
    os.makedirs(output_dir, exist_ok=True)

    input_top = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.top"
    input_dat = "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.dat"

    # Parameters for the mutation actions
    mutation_percentage = 0.5
    mutation_zone_width = 0.10
    num_mutations = 5

    # --- Test 1: Deletion ---
    print(f"\n[1/2] Loading structure for DELETION test...")
    try:
        dna_del = DNAStructure(top_path=input_top, dat_path=input_dat)
        initial_count = len(dna_del.nucleotides)
        print(f"Initial nucleotide count: {initial_count}")

        success = random_deletion_in_zone(dna_del, mutation_percentage, num_mutations, mutation_zone_width)
        
        if success:
            print(f"Final nucleotide count: {len(dna_del.nucleotides)}")
            del_output_top = os.path.join(output_dir, "deleted.top")
            del_output_dat = os.path.join(output_dir, "deleted.dat")
            # Use the robust re-ordering save method
            dna_del.save_files_after_ligation(del_output_top, del_output_dat)

    except Exception as e:
        print(f"❌ Failed during deletion test. Error: {e}")

    # --- Test 2: Insertion ---
    print(f"\n[2/2] Loading structure for INSERTION test...")
    try:
        dna_ins = DNAStructure(top_path=input_top, dat_path=input_dat)
        initial_count = len(dna_ins.nucleotides)
        print(f"Initial nucleotide count: {initial_count}")

        success = random_insertion_in_zone(dna_ins, mutation_percentage, num_mutations, mutation_zone_width)
        
        if success:
            print(f"Final nucleotide count: {len(dna_ins.nucleotides)}")
            ins_output_top = os.path.join(output_dir, "inserted.top")
            ins_output_dat = os.path.join(output_dir, "inserted.dat")
            # Use the robust re-ordering save method
            dna_ins.save_files_after_ligation(ins_output_top, ins_output_dat)
            
    except Exception as e:
        print(f"❌ Failed during insertion test. Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting action test script...")
    
    # === CHOOSE WHICH TEST TO RUN ===
    # Uncomment the function you want to execute.
    
    # test_nicking()
    # test_nick_then_ligate()
    # test_ligate_only()
    # test_induce_bend()
    test_zone_mutations()

    print("\n✅ Test script finished.")
