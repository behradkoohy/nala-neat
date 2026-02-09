import hydra
import ray
import time
import torch
from omegaconf import DictConfig
from src.island import Island
import gymnasium as gym
import torch
import time


def watch_champion(genome):
    # 'render_mode="human"' creates the pop-up window
    env = gym.make("CartPole-v1", render_mode="human")

    state, _ = env.reset()
    total_reward = 0
    done = False

    print("--- Watching Champion ---")
    while not done:
        # 1. Forward Pass
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = genome(state_t)
            action = torch.argmax(logits, dim=1).item()

        # 2. Step
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # 3. Slow down so you can see it
        time.sleep(0.02)

    print(f"Final Score: {total_reward}")
    env.close()

def calculate_diversity(champions):
    if not champions or None in champions: return 0.0

    # Check type of first champion
    if hasattr(champions[0], 'connections'):
        # --- GRAPH GENOME METRIC ---
        # Measure diversity by "Connection Overlap" (Jaccard Distance)
        scores = []
        for i in range(len(champions)):
            for j in range(i + 1, len(champions)):
                c1 = set(champions[i].connections.keys())
                c2 = set(champions[j].connections.keys())
                intersection = len(c1.intersection(c2))
                union = len(c1.union(c2))
                jaccard = 1.0 - (intersection / union) if union > 0 else 0.0
                scores.append(jaccard)
        return sum(scores) / len(scores) if scores else 0.0

    else:
        # --- DENSE GENOME METRIC (Existing) ---
        params_list = []
        for champ in champions:
            vec = torch.cat([p.view(-1) for p in champ.parameters()])
            params_list.append(vec)
        stack = torch.stack(params_list)
        mean = stack.mean(dim=0)
        dist = (stack - mean).pow(2).sum(dim=1).mean().item()
        return dist


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"--> Spawning {cfg.num_islands} islands for task: {cfg.task._target_}")

    # Spawn Islands (Actors)
    islands = [Island.remote(i, cfg) for i in range(cfg.num_islands)]

    try:
        while True:
            time.sleep(cfg.poll_interval)

            # 1. Gather Champions
            futures = [island.get_champion.remote() for island in islands]
            champions = ray.get(futures)

            # 2. Filter valid champions (ignore startups)
            valid_champs = [c for c in champions if c is not None]
            if not valid_champs:
                print("Waiting for first generation...")
                continue

            # 3. Metrics & Logging
            div_score = calculate_diversity(valid_champs)
            best_fitness = max(c.fitness for c in valid_champs)
            print(f"Global Diversity: {div_score:.5f} | Best Fitness: {best_fitness:.4f}")

            """# 4. Migration (Ring Topology)
            # Sends Champ[0] -> Island[1], Champ[1] -> Island[2], etc.
            if len(valid_champs) == cfg.num_islands:
                for i in range(len(islands)):
                    source_champ = valid_champs[i]
                    target_island = islands[(i + 1) % len(islands)]
                    target_island.receive_migrant.remote(source_champ)
            """
            # 4. Migration (filtered)
            if len(valid_champs) == cfg.num_islands:
                for i in range(len(islands)):
                    source_champ = valid_champs[i]
                    target_island = islands[(i + 1) % len(islands)]

                    target_champ = valid_champs[(i + 1) % len(islands)]
                    if target_champ.fitness < 3.9:
                        target_island.receive_migrant.remote(source_champ)

            # Debug: used for checking champion in cartpole. could be extended.
            # watch_champion(champions[0])

    except KeyboardInterrupt:
        print("\nStopping Evolution...")
        ray.shutdown()


if __name__ == "__main__":
    main()