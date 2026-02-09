import torch
import math
import gymnasium as gym  # pip install gymnasium

class XORTask:
    """
    A simple task that matches the interface expected by the Island.
    """
    def __init__(self, input_size=2, output_size=1):
        self.input_size = input_size
        self.output_size = output_size
        # XOR Data
        self.inputs = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
        self.targets = torch.tensor([[0.], [1.], [1.], [0.]])

    def evaluate(self, genome):
        """
        Takes a genome (PyTorch Module), returns a float fitness.
        """
        with torch.no_grad():
            preds = genome(self.inputs)
            loss = ((preds - self.targets) ** 2).sum()
            return 4.0 - loss.item() # Max fitness = 4.0


class RastriginTask:
    """
    Goal: Find x = [0, 0, ...], where f(x) = 0.
    We ask the network: "Given an input of all 1s, what vector x do you output?"
    """

    def __init__(self, input_size=10, output_size=1):
        self.input_size = input_size
        # CHANGE: Output size must match input size so the network outputs the full vector 'x'
        self.output_size = input_size

    def evaluate(self, genome):
        # 1. Create a dummy input (vector of 1s)
        # We need a batch dimension for safety: (1, input_size)
        dummy_input = torch.ones(1, self.input_size)

        # 2. Get the output 'x' from the genome
        # (This works for both NeuroGenome and GraphGenome)
        with torch.no_grad():
            output = genome(dummy_input)

        # Flatten to 1D vector
        x = output.view(-1)

        # Scale to domain [-5.12, 5.12] (assuming tanh output is -1 to 1)
        x = x * 5.12

        # 3. Calculate Rastrigin value
        # f(x) = 10n + sum(x^2 - 10cos(2pi*x))
        n = self.input_size
        score = 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * math.pi * x))

        # 4. Fitness = Minimize Score (return negative)
        return -score.item()


class CartPoleTask:
    """
    Classic Control Theory.
    Goal: Balance a pole on a cart.
    Input: [Cart Pos, Cart Vel, Pole Angle, Pole Vel]
    Output: [Push Left, Push Right] (Binary)
    """

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.input_size = 4
        self.output_size = 2  # Softmax prob for Left/Right

    def evaluate(self, genome):
        state, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            # Convert state to tensor
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                logits = genome(state_t)
                action = torch.argmax(logits, dim=1).item()

            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            if total_reward >= 500:  # Solved
                break

        return total_reward


class MountainCarTask:
    """
    The 'Momentum' Test.
    Goal: Drive a weak car up a steep hill.
    Challenge: The car is too weak to drive straight up. It must rock back and forth
    to build momentum.
    """

    def __init__(self):
        # 'MountainCar-v0' is the standard version
        self.env = gym.make("MountainCar-v0")
        self.input_size = 2  # [Position, Velocity]
        self.output_size = 3  # [Left, Neutral, Right]

    def evaluate(self, genome):
        state, _ = self.env.reset()
        done = False
        truncated = False

        max_height = -1.2  # Start at bottom
        step_count = 0

        while not (done or truncated):
            # Convert state to tensor for the network
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                logits = genome(state_t)
                action = torch.argmax(logits, dim=1).item()

            # Step the environment
            state, reward, done, truncated, _ = self.env.step(action)

            # Track progress
            if state[0] > max_height:
                max_height = state[0]
            step_count += 1

            # Early stopping if it's just sitting at the bottom (optional optimization)
            if step_count > 200:
                truncated = True

        # --- FITNESS CALCULATION ---
        # The goal position is 0.5.

        if max_height >= 0.5:
            # SUCCESS: Reward is based on efficiency (fewer steps is better)
            # Max steps is 200, so we give a bonus + the steps saved.
            return 200.0 - step_count
        else:
            # FAILURE: Return max height as a "hint" to guide evolution.
            # Without this, fitness is flat -200 for everyone.
            # We scale it to be negative but distinct (e.g., -1.2 to 0.5).
            return max_height