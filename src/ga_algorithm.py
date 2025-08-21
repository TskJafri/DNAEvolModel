# src/ga_algorithm.py

import os
import random
import shutil
import subprocess
import sys
import argparse
import numpy as np
from typing import List, Optional, Dict, Any, Callable

# --- Project Path Setup ---
# Ensures that modules like dna_model can be found
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dna_model import DNAStructure
from src.fitness import FitnessCalculator
from src.target_scaffold import TargetScaffold
from src.oop_functions import (
    random_mutate,
    target_nick,
    truncate_strand,
    random_deletion_in_zone,
    random_insertion_in_zone
)

# --- GA Individual ---
class Individual:
    """Represents a single DNA structure in the population."""
    def __init__(self, generation: int, individual_id: int, base_dir: str):
        self.generation = generation
        self.id = individual_id
        self.fitness = float('inf')
        self.is_evaluated = False
        
        # File management
        self.dir_path = os.path.join(base_dir, f"gen_{generation}", f"id_{individual_id}")
        os.makedirs(self.dir_path, exist_ok=True)
        self.top_path = os.path.join(self.dir_path, "structure.top")
        self.dat_path = os.path.join(self.dir_path, "structure.dat")

    def __repr__(self):
        return f"Individual(gen={self.generation}, id={self.id}, fitness={self.fitness:.4f})"

# --- The Genetic Algorithm Engine ---
class GeneticAlgorithm:
    """Manages the evolutionary process."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population: List[Individual] = []
        
        print("🧬 Initializing Genetic Algorithm...")
        self.target_scaffold = TargetScaffold(config["target_xyz"])
        self.fitness_calculator = FitnessCalculator(
            target_scaffold=self.target_scaffold,
            weight_shape=config["weight_shape"],
            weight_integrity=config["weight_integrity"]
        )
        
        # Clean up previous runs
        if os.path.exists(config["work_dir"]):
            shutil.rmtree(config["work_dir"])
        os.makedirs(config["work_dir"])

        # Adaptive mutation tracking
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
        self.fitness_history = []
        
        # Store base mutation rates for adaptive adjustment
        self.base_mutation_rates = {
            "guided_deletion": 0.35,
            "random_deletion": 0.20,
            "random_insertion": 0.20,
            "random_base_mutate": 0.30,
            "random_nick": 0.15,
            "random_truncate": 0.15
        }

        # Define the mutation operators and their probabilities (increased from original)
        self.mutation_operators: List[Dict[str, Any]] = [
            {"func": self._guided_deletion, "prob": 0.35, "name": "Guided Deletion"},
            {"func": self._random_deletion, "prob": 0.20, "name": "Random Deletion"},
            {"func": self._random_insertion, "prob": 0.20, "name": "Random Insertion"},
            {"func": self._random_base_mutate, "prob": 0.30, "name": "Random Base Mutation"},
            {"func": self._random_nick, "prob": 0.15, "name": "Random Nick"},
            {"func": self._random_truncate, "prob": 0.15, "name": "Random Truncation"}
        ]

    def _run_relaxation(self, individual: Individual) -> bool:
        """Executes the external relaxation script. Returns True on success, False on failure."""
        print(f"    -> Running relaxation for Individual (Gen {individual.generation}, ID {individual.id})...")
        relax_script_path = os.path.join(PROJECT_ROOT, self.config["relax_script"])
        python_executable = sys.executable
        command = [python_executable, relax_script_path, individual.dir_path]
        
        try:
            shutil.move(individual.top_path, os.path.join(individual.dir_path, "input.top"))
            shutil.move(individual.dat_path, os.path.join(individual.dir_path, "last_conf.dat"))

            subprocess.run(command, check=True, capture_output=True, text=True)
            
            relaxed_dat = os.path.join(individual.dir_path, "md_relax", "last_conf.dat")
            if not os.path.exists(relaxed_dat):
                raise FileNotFoundError("Relaxation finished, but output .dat file not found.")

            shutil.copy(os.path.join(individual.dir_path, "md_relax", "input.top"), individual.top_path)
            shutil.copy(relaxed_dat, individual.dat_path)
            return True

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    -> ❌ ERROR: Relaxation failed for Individual {individual.id}.")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"    -> STDOUT: {e.stdout}")
                print(f"    -> STDERR: {e.stderr}")
            return False
            
        finally:
            if os.path.exists(os.path.join(individual.dir_path, "input.top")):
                 os.remove(os.path.join(individual.dir_path, "input.top"))
            if os.path.exists(os.path.join(individual.dir_path, "last_conf.dat")):
                 os.remove(os.path.join(individual.dir_path, "last_conf.dat"))

    def _evaluate_individual(self, individual: Individual):
        """Calculates the fitness of a single individual."""
        if individual.is_evaluated:
            return

        relaxation_succeeded = self._run_relaxation(individual)
        
        if not relaxation_succeeded:
            individual.fitness = float('inf')
            individual.is_evaluated = True
            return

        print(f"    -> Evaluating fitness for Individual (Gen {individual.generation}, ID {individual.id})...")
        try:
            dna = DNAStructure(individual.top_path, individual.dat_path)
            individual.fitness = self.fitness_calculator.calculate_fitness(dna)
        except (ValueError, FileNotFoundError) as e:
            print(f"    -> ⚠️ WARNING: Could not load/evaluate individual {individual.id}. Reason: {e}")
            individual.fitness = float('inf')
        
        individual.is_evaluated = True
        print(f"    -> Fitness = {individual.fitness:.4f}")

    def _initialize_population(self):
        """Creates the starting population from the initial structure."""
        print("\n--- Initializing Generation 0 ---")
        initial_dna = DNAStructure(self.config["initial_top"], self.config["initial_dat"])
        for i in range(self.config["pop_size"]):
            ind = Individual(generation=0, individual_id=i, base_dir=self.config["work_dir"])
            initial_dna.save_files_minimal_change(ind.top_path, ind.dat_path)
            self.population.append(ind)
            self._evaluate_individual(ind)
        
        self.population.sort(key=lambda x: x.fitness)
        print("--- Initialization complete ---")

    def _adapt_mutation_rates(self, current_best_fitness: float):
        """Dynamically adjust mutation rates based on progress."""
        improvement = self.last_best_fitness - current_best_fitness
        
        # Check for stagnation
        if improvement < 0.005:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Apply adaptive mutation strategy
        if self.stagnation_counter >= 3:  # More aggressive trigger
            boost_factor = min(2.0, 1.0 + (self.stagnation_counter - 2) * 0.2)  # 20% increase per stagnant gen
            print(f"  🔥 Stagnation detected ({self.stagnation_counter} gens)! Boosting mutation rates by {boost_factor:.1f}x")
            
            # Update mutation probabilities
            for i, op in enumerate(self.mutation_operators):
                base_rate = list(self.base_mutation_rates.values())[i]
                new_rate = min(base_rate * boost_factor, 0.85)  # Cap at 85%
                op["prob"] = new_rate
                
        elif self.stagnation_counter == 0 and improvement > 0.01:
            # Good progress - reset to base rates
            for i, op in enumerate(self.mutation_operators):
                base_rate = list(self.base_mutation_rates.values())[i]
                op["prob"] = base_rate
        
        self.last_best_fitness = current_best_fitness

    def _select_parents(self) -> List[Individual]:
        """Selects parents for the next generation using tournament selection."""
        parents = []
        for _ in range(self.config["pop_size"]):
            tournament = random.sample(self.population, self.config["tournament_size"])
            winner = min(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

    def _apply_mutations(self, dna: DNAStructure, parent: Individual) -> bool:
        """Applies a series of probabilistic mutations to a DNA structure."""
        mutated = False
        mutations_applied = []
        
        print(f"    -> Mutating child from parent {parent.id}...")
        for op in self.mutation_operators:
            if random.random() < op["prob"]:
                print(f"       - Applying: {op['name']} (prob={op['prob']:.2f})")
                success = op["func"](dna, parent) 
                if success:
                    mutated = True
                    mutations_applied.append(op['name'])
        
        if not mutated:
            print("       - No mutations were applied this time.")
        else:
            print(f"       - Successfully applied: {', '.join(mutations_applied)}")
        
        return mutated

    # --- Wrapper methods for mutation operators ---
    def _guided_deletion(self, dna: DNAStructure, parent: Individual) -> bool:
        """Deletes nucleotides from the area with the worst deviation."""
        parent_dna = DNAStructure(parent.top_path, parent.dat_path)
        analysis = self.fitness_calculator.find_worst_deviation(parent_dna)
        if analysis:
            percentage, _ = analysis
            return random_deletion_in_zone(dna, percentage=percentage, num_deletions=2, zone_width=0.15)
        return False

    def _random_deletion(self, dna: DNAStructure, parent: Individual) -> bool:
        return random_deletion_in_zone(dna, percentage=random.random(), num_deletions=1, zone_width=0.10)
    
    def _random_insertion(self, dna: DNAStructure, parent: Individual) -> bool:
        return random_insertion_in_zone(dna, percentage=random.random(), num_insertions=1, zone_width=0.10)
    
    def _random_base_mutate(self, dna: DNAStructure, parent: Individual) -> bool:
        """Applies a random base pair mutation."""
        success = random_mutate(dna)
        return success is True

    def _random_nick(self, dna: DNAStructure, parent: Individual) -> bool:
        if len(dna.nucleotides) > 1:
            nick_idx = random.randint(0, len(dna.nucleotides) - 2)
            return target_nick(dna, nick_after_idx=nick_idx)
        return False

    def _random_truncate(self, dna: DNAStructure, parent: Individual) -> bool:
        strands = list(set(n.strand_id for n in dna.nucleotides))
        if not strands: return False
        strand_id = random.choice(strands)
        end = random.choice(['5_prime', '3_prime'])
        return truncate_strand(dna, strand_id=strand_id, end_to_truncate=end, num_to_truncate=1)

    def run(self):
        """The main loop of the genetic algorithm."""
        self._initialize_population()
        if not self.population or self.population[0].fitness == float('inf'):
            print("❌ FATAL: Initial population is not viable. Aborting run.")
            return

        best_overall_fitness = self.population[0].fitness
        self.fitness_history.append(best_overall_fitness)

        for gen in range(1, self.config["generations"] + 1):
            print(f"\n{'='*15} Generation {gen} {'='*15}")
            
            self.population.sort(key=lambda x: x.fitness)
            best_in_gen = self.population[0]
            
            if best_in_gen.fitness == float('inf'):
                print("💀 Population has entered a death spiral. All individuals are invalid. Aborting.")
                break
            
            print(f"  Best fitness in Gen {gen-1}: {best_in_gen.fitness:.4f}")
            
            # Track improvements
            if best_in_gen.fitness < best_overall_fitness:
                improvement = best_overall_fitness - best_in_gen.fitness
                best_overall_fitness = best_in_gen.fitness
                print(f"  🎉 New best overall fitness! Improvement: {improvement:.4f}")
            
            # Adaptive mutation strategy
            self._adapt_mutation_rates(best_in_gen.fitness)
            self.fitness_history.append(best_in_gen.fitness)
            
            # Progress reporting every 10 generations
            if gen % 10 == 0:
                recent_change = self.fitness_history[-10] - self.fitness_history[-1] if len(self.fitness_history) >= 10 else 0
                print(f"  📊 Last 10 gen improvement: {recent_change:.4f}")
                current_rates = [f"{op['name']}: {op['prob']:.2f}" for op in self.mutation_operators]
                print(f"  🎯 Current mutation rates: {', '.join(current_rates[:3])}")

            # Create viable mating pool
            mating_pool = [ind for ind in self.population if ind.fitness != float('inf')]
            
            if not mating_pool:
                print("💀 No viable individuals left to act as parents. Aborting.")
                break

            print(f"  -> Viable individuals for mating pool: {len(mating_pool)}/{len(self.population)}")

            # Create the next generation
            next_generation = []
            
            # Elitism: Carry over the best individuals
            elite_count = min(self.config["elitism_count"], len(mating_pool))
            elites = mating_pool[:elite_count]
            for i, elite in enumerate(elites):
                child = Individual(generation=gen, individual_id=i, base_dir=self.config["work_dir"])
                shutil.copy(elite.top_path, child.top_path)
                shutil.copy(elite.dat_path, child.dat_path)
                child.fitness = elite.fitness
                child.is_evaluated = True 
                next_generation.append(child)
            print(f"  -> Preserving {len(elites)} elite individuals.")

            # Create new children from the safe mating pool
            num_children = self.config["pop_size"] - len(elites)
            for i in range(num_children):
                tournament_contenders = random.sample(mating_pool, k=min(self.config["tournament_size"], len(mating_pool)))
                parent = min(tournament_contenders, key=lambda x: x.fitness)

                child = Individual(generation=gen, individual_id=i + len(elites), base_dir=self.config["work_dir"])
                
                dna_to_mutate = DNAStructure(parent.top_path, parent.dat_path)
                self._apply_mutations(dna_to_mutate, parent)
                dna_to_mutate.save_files_minimal_change(child.top_path, child.dat_path)
                self._evaluate_individual(child)
                next_generation.append(child)
            
            self.population = next_generation

        # Final Results
        print("\n\n--- Optimization Run Finished ---")
        self.population.sort(key=lambda x: x.fitness)
        if self.population:
            best_final = self.population[0]
            print(f"Best structure found in Generation {best_final.generation} (ID: {best_final.id})")
            print(f"Final Fitness: {best_final.fitness:.4f}")
            print(f"Final files are in directory: {best_final.dir_path}")
            
            # Print fitness history summary
            if len(self.fitness_history) > 1:
                total_improvement = self.fitness_history[0] - self.fitness_history[-1]
                print(f"Total improvement: {total_improvement:.4f} over {len(self.fitness_history)-1} generations")
        else:
            print("No valid individuals remained at the end of the run.")

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Evolve a DNA nanostructure using a Genetic Algorithm.")
    
    # File Paths
    parser.add_argument('--initial_top', type=str, required=True, help="Path to the initial topology (.top) file.")
    parser.add_argument('--initial_dat', type=str, required=True, help="Path to the initial coordinate (.dat) file.")
    parser.add_argument('--target_xyz', type=str, required=True, help="Path to the target shape (.xyz) file.")
    parser.add_argument('--work_dir', type=str, default="ga_run", help="Directory to store generations and results.")
    parser.add_argument('--relax_script', type=str, default="src/utils/relax.py", help="Path to the relaxation script.")

    # GA Parameters
    parser.add_argument('--generations', type=int, default=50, help="Number of generations to run.")
    parser.add_argument('--pop_size', type=int, default=20, help="Number of individuals in the population.")
    parser.add_argument('--tournament_size', type=int, default=3, help="Number of individuals in a selection tournament.")
    parser.add_argument('--elitism_count', type=int, default=2, help="Number of best individuals to carry to the next generation.")

    # Fitness Weights
    parser.add_argument('--weight_shape', type=float, default=1.0, help="Weight for the shape fitness component (f_shape).")
    parser.add_argument('--weight_integrity', type=float, default=0.1, help="Weight for the integrity fitness component (f_integrity).")

    args = parser.parse_args()
    config = vars(args)

    ga = GeneticAlgorithm(config)
    ga.run()

if __name__ == "__main__":
    main()