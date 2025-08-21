# scripts/relax.py

from ipy_oxdna.oxdna_simulation import Simulation
from pathlib import Path
import multiprocessing as mp
import sys

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

    # --- FIX: Get the working directory from the GA's command line argument ---
    if len(sys.argv) < 2:
        print("❌ ERROR: Please provide the path to the individual's working directory.")
        print("   Usage: python relax.py /path/to/ga_run/gen_X/id_Y")
        sys.exit(1)
        
    working_dir = Path(sys.argv[1])
    # --- END FIX ---

    # Your original, clean logic begins here.
    # The simulation will auto-detect the input.top and last_conf.dat files.
    
    print(f"🧬 Starting MC relaxation on evolved structure in {working_dir}...")
    relax_mc = Simulation(working_dir, working_dir / "mc_relax")
    relax_mc.build()
    relax_mc.input["steps"] = 500
    relax_mc.input["print_conf_interval"] = 500
    relax_mc.input["print_energy_interval"] = 500
    relax_mc.input.swap_default_input("cpu_MC_relax")
    relax_mc.input["T"] = "30C"
    relax_mc.input["box_type"] = "orthogonal"
    relax_mc.oxpy_run(join=True)
    print("✓ MC relaxation completed!")

    print("🧬 Starting MD relaxation...")
    relax_md = Simulation(working_dir / "mc_relax", working_dir / "md_relax")
    relax_md.build()
    relax_md.input.swap_default_input("cpu_MD_relax")
    relax_md.input["steps"] = 1000
    relax_md.input["print_conf_interval"] = 500
    relax_md.input["print_energy_interval"] = 500
    relax_md.input["print_energy_every"] = 500
    relax_md.input["T"] = "30C"
    relax_md.oat.generate_force()
    relax_md.input["box_type"] = "orthogonal"
    relax_md.oxpy_run(join=True)
    print("✓ MD relaxation completed!")
    
    print("\n🎉 Complete relaxation pipeline finished!")
    print(f"📁 Results in: {working_dir}/mc_relax/ and {working_dir}/md_relax/")