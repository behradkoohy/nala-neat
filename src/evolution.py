import torch
import torch.nn as nn
import copy
import random


class GraphGenome:
    """
    A True Sparse Network.
    Nodes are integers:
      0..input-1 (Inputs)
      input..input+output-1 (Outputs)
      input+output..inf (Hidden)
    """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.fitness = -float('inf')

        # Structure: {(source_node, target_node): weight}
        self.connections = {}

        # Node Biases: {node_id: bias_value}
        self.biases = {}

        # Track max node ID to add new ones safely
        self.max_node = input_size + output_size - 1

        # Initialize Output Biases
        for i in range(output_size):
            out_node = input_size + i
            self.biases[out_node] = 0.0

        # Initialize dense connection from Input -> Output
        for i in range(input_size):
            for j in range(output_size):
                out_node = input_size + j
                # Random weight initialization
                self.connections[(i, out_node)] = torch.randn(1).item()

    # --- FIX 1: Add __call__ to make usage 'genome(x)' valid ---
    def __call__(self, x):
        return self.forward(x)

    # --- FIX 2: Handle Batches (dim > 1) ---
    def forward(self, x):
        # Handle Batch Logic (XORTask sends [4, 2], RL tasks send [1, 4])
        if x.dim() > 1:
            # Loop through the batch and stack results
            batch_outputs = [self._forward_single(x[i]) for i in range(x.size(0))]
            return torch.stack(batch_outputs)
        else:
            # Handle single input
            return self._forward_single(x)

    def _forward_single(self, x):
        # 1. Reset Node Values
        node_values = {}

        # 2. Set Inputs
        for i in range(self.input_size):
            node_values[i] = x[i].item()

        # 3. Simple Feed-Forward Logic
        outputs = []
        for i in range(self.output_size):
            out_node = self.input_size + i
            val = self._compute_node(out_node, node_values, visited=set())
            outputs.append(val)

        return torch.tensor(outputs)

    def _compute_node(self, node_id, values, visited):
        """Recursive calculation with loop detection."""
        if node_id in values: return values[node_id]
        if node_id in visited: return 0.0  # Break cycles (Recurrence treated as 0 this step)

        visited.add(node_id)

        # Sum incoming connections
        total = self.biases.get(node_id, 0.0)

        # Find all connections pointing TO this node
        incoming = [k for k in self.connections.keys() if k[1] == node_id]

        for (src, dst) in incoming:
            weight = self.connections[(src, dst)]
            val = self._compute_node(src, values, visited)
            total += val * weight

        # Activation (Tanh)
        activation = torch.tanh(torch.tensor(total)).item()
        values[node_id] = activation
        return activation

    # --- MUTATION OPERATORS ---

    def mutate(self, power, rate):
        r = random.random()

        if r < 0.1:
            self._mutate_add_node()
        elif r < 0.2:
            self._mutate_add_connection()
        else:
            self._mutate_weights(power)

    def _mutate_weights(self, power):
        # Mutate connections
        for k in self.connections:
            if random.random() < 0.5:
                self.connections[k] += torch.randn(1).item() * power
        # Mutate biases
        for k in self.biases:
            if random.random() < 0.5:
                self.biases[k] += torch.randn(1).item() * power

    def _mutate_add_connection(self):
        # Pick two random existing nodes
        src = random.randint(0, self.max_node)
        dst = random.randint(self.input_size, self.max_node)  # Don't connect TO inputs

        if src == dst: return
        if (src, dst) in self.connections: return

        self.connections[(src, dst)] = torch.randn(1).item()

    def _mutate_add_node(self):
        if not self.connections: return

        # 1. Pick a connection to split
        conn_key = random.choice(list(self.connections.keys()))
        src, dst = conn_key
        old_weight = self.connections.pop(conn_key)

        # 2. Create new node
        self.max_node += 1
        new_node = self.max_node
        self.biases[new_node] = 0.0

        # 3. Add links: src -> new -> dst
        self.connections[(src, new_node)] = 1.0
        self.connections[(new_node, dst)] = old_weight

    @staticmethod
    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)  # Start with Parent 1 topology

        # Probabilistically add Parent 2's unique features
        for conn, weight in parent2.connections.items():
            if conn not in child.connections:
                if random.random() < 0.5:
                    child.connections[conn] = weight
            else:
                child.connections[conn] = (child.connections[conn] + weight) / 2.0

        for node, bias in parent2.biases.items():
            if node not in child.biases:
                child.biases[node] = bias

        child.max_node = max(parent1.max_node, parent2.max_node)
        return child


class NeuroGenome(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 1. The "DNA" (Weights)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

        # 2. The "Topology" (Masks)
        self.mask1 = torch.ones_like(self.l1.weight)
        self.mask2 = torch.ones_like(self.l2.weight)

        # Turn off gradients
        for param in self.parameters():
            param.requires_grad = False
        self.fitness = -float('inf')

    def forward(self, x):
        w1 = self.l1.weight * self.mask1
        x = torch.relu(torch.nn.functional.linear(x, w1, self.l1.bias))
        w2 = self.l2.weight * self.mask2
        x = torch.sigmoid(torch.nn.functional.linear(x, w2, self.l2.bias))
        return x

    # --- MUTATION OPERATORS ---
    def mutate_weight_sparse(self, power):
        all_params = [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias]
        target = random.choice(all_params)
        idx = random.randint(0, target.numel() - 1)
        with torch.no_grad():
            target.view(-1)[idx] += torch.randn(1).item() * power

    def mutate_topology_toggle(self):
        target_mask = random.choice([self.mask1, self.mask2])
        idx = random.randint(0, target_mask.numel() - 1)
        current = target_mask.view(-1)[idx].item()
        target_mask.view(-1)[idx] = 1.0 if current == 0.0 else 0.0

    @staticmethod
    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for p_c, p2 in zip(child.parameters(), parent2.parameters()):
                if random.random() < 0.5:
                    p_c.data.copy_(p2.data)
            if random.random() < 0.5:
                child.mask1.copy_(parent2.mask1)
            if random.random() < 0.5:
                child.mask2.copy_(parent2.mask2)
        return child


class Population:
    def __init__(self, cfg, input_size, output_size):
        self.cfg = cfg
        self.input_size = input_size
        self.output_size = output_size

        # FACTORY SWITCH
        genome_type = getattr(cfg, "genome_type", "dense")  # Default to dense

        self.individuals = []
        for _ in range(cfg.pop_size):
            if genome_type == "graph":
                ind = GraphGenome(input_size, output_size)
            else:
                ind = NeuroGenome(input_size, cfg.hidden_size, output_size)
            self.individuals.append(ind)

        self.generation = 0
        self.genome_class = GraphGenome if genome_type == "graph" else NeuroGenome

    def get_best(self):
        return max(self.individuals, key=lambda ind: ind.fitness)

    def evolve_step(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        elites = self.individuals[:self.cfg.elitism_count]

        next_gen = [copy.deepcopy(e) for e in elites]

        while len(next_gen) < self.cfg.pop_size:
            parent1 = random.choice(self.individuals[:15])
            parent2 = random.choice(self.individuals[:15])

            child = self.genome_class.crossover(parent1, parent2)
            child.mutate(self.cfg.mutation_power, self.cfg.mutation_rate)

            next_gen.append(child)

        self.individuals = next_gen
        self.generation += 1

    def inject_migrant(self, migrant_genome):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals[-1] = copy.deepcopy(migrant_genome)

    def trigger_cataclysm(self):
        """Keeps the best 1, randomizes everyone else."""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        best = self.individuals[0]

        # Replace everyone else with fresh random genomes
        new_pop = [best]
        for _ in range(self.cfg.pop_size - 1):
            if self.genome_class == GraphGenome:
                new_pop.append(GraphGenome(self.input_size, self.output_size))
            else:
                new_pop.append(NeuroGenome(self.input_size, self.cfg.hidden_size, self.output_size))

        self.individuals = new_pop