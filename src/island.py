import ray
import threading
import time
import copy
import torch
from hydra.utils import instantiate
from src.evolution import Population


@ray.remote
class Island:
    def __init__(self, island_id, cfg):
        self.island_id = island_id
        self.cfg = cfg
        self.task = instantiate(cfg.task)

        self.population = Population(
            cfg.ga,
            input_size=self.task.input_size,
            output_size=self.task.output_size
        )

        self.shared_state = {
            "champion": None,
            "gen": 0,
            "lock": threading.Lock()
        }
        self.migrant_queue = []

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        stagnation_counter = 0
        last_best_fitness = -float('inf')

        try:
            while self.running:
                # 1. Check for Migrants
                if self.migrant_queue:
                    migrant = self.migrant_queue.pop(0)
                    self.population.inject_migrant(migrant)

                # 2. Evaluate Fitness
                for ind in self.population.individuals:
                    ind.fitness = self.task.evaluate(ind)

                # 3. Update Stats
                champion = self.population.get_best()
                if champion.fitness > last_best_fitness + 0.01:
                    # Improvement! Reset counter
                    stagnation_counter = 0
                    last_best_fitness = champion.fitness
                else:
                    # No improvement
                    stagnation_counter += 1

                    # TRIGGER CATACLYSM
                    # If stuck for ~50 generations (0.5s), nuke the population
                if stagnation_counter > 50:
                    print(f"Island {self.island_id} STAGNATED. Triggering Cataclysm!")
                    self.population.trigger_cataclysm()  # <--- You need to add this method
                    stagnation_counter = 0

                with self.shared_state["lock"]:
                    self.shared_state["champion"] = copy.deepcopy(champion)
                    self.shared_state["gen"] = self.population.generation



                # 4. Evolve
                self.population.evolve_step()
                time.sleep(0.01)

        except Exception as e:
            print(f"!!! ISLAND {self.island_id} CRASHED !!!")
            print(e)
            import traceback
            traceback.print_exc()

    def get_champion(self):
        with self.shared_state["lock"]:
            return self.shared_state["champion"]

    def receive_migrant(self, genome):
        # 1. Calculate distance between Migrant and Local Champion
        with self.shared_state["lock"]:
            current_champ = self.shared_state["champion"]

        distance = 100.0  # Default to "very different" if we can't compare

        if current_champ is not None:
            # --- CASE A: Dense Genome (NeuroGenome) ---
            if hasattr(genome, 'parameters'):
                w1 = torch.cat([p.view(-1) for p in genome.parameters()])
                w2 = torch.cat([p.view(-1) for p in current_champ.parameters()])
                distance = torch.norm(w1 - w2).item()

            # --- CASE B: Sparse Genome (GraphGenome) ---
            elif hasattr(genome, 'connections'):
                # 1. Check Topology Differences (Keys)
                k1 = set(genome.connections.keys())
                k2 = set(current_champ.connections.keys())

                if k1 != k2:
                    # If structures differ, they are definitely distinct
                    distance = 50.0
                else:
                    # If structures match, check Weight differences
                    diff = 0.0
                    for k in k1:
                        diff += (genome.connections[k] - current_champ.connections[k]) ** 2
                    distance = diff ** 0.5

        # 2. Reject if too similar
        if distance < 0.1:
            return  # REJECT CLONE

        # If distinct, add to queue
        clone = copy.deepcopy(genome)
        self.migrant_queue.append(clone)