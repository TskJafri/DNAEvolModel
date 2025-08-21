from random import random

from matplotlib.pylab import trunc
from src.dna_model import DNAStructure, Nucleotide
from src.oop_functions import target_ligate, target_nick, target_mutate, random_mutate, remove_random_strand, truncate_strand

def run_ligation_simulation():
    TOP_IN = "/Users/taskeenjafri/Projects/EvolModel/short.top"
    DAT_IN = "/Users/taskeenjafri/Projects/EvolModel/short.dat"

    TOP_OUT = "/Users/taskeenjafri/Projects/EvolModel/newligated.top"
    DAT_OUT = "/Users/taskeenjafri/Projects/EvolModel/newligated.dat"

    # 1. Load both files into a single, unified data structure.
    dna = DNAStructure(top_path=TOP_IN, dat_path=DAT_IN)

    success = target_nick(dna, nick_after_idx=156)

    if success:
        dna.save_files_minimal_change(out_top_path=TOP_OUT, out_dat_path=DAT_OUT)

    # 2. Perform the targeted action in memory with explicit ends.
    # This connects the 3' end of strand 5 to the 5' end of strand 13.
    # success = target_ligate(dna, 
    #                         strand1_id=5, strand1_end='5_prime', 
    #                         strand2_id=13, strand2_end='3_prime')

    # 3. If the action was successful, save the new state to both files.
    if success:
        dna.save_files_minimal_change(out_top_path=TOP_OUT, out_dat_path=DAT_OUT)

# if __name__ == "__main__":
#     run_ligation_simulation()

TOP_IN = "/Users/taskeenjafri/Projects/EvolModel/short.top"
DAT_IN = "/Users/taskeenjafri/Projects/EvolModel/short.dat"

TOP_OUT = "/Users/taskeenjafri/Projects/EvolModel/newligated2.top"
DAT_OUT = "/Users/taskeenjafri/Projects/EvolModel/newligated2.dat"

dna = DNAStructure(top_path=TOP_IN, dat_path=DAT_IN)

success = truncate_strand(dna, strand_id=5, end_to_truncate='3_prime', num_to_truncate=10)

if success:
    dna.save_files_minimal_change(out_top_path=TOP_OUT, out_dat_path=DAT_OUT)

print("\nSimulation complete. Files saved if successful.")