import gym_env

import pickle
import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

from collections import deque, namedtuple
from itertools import count
from torch import nn, optim
from torchrl.data import ReplayBuffer, ListStorage

# Define gym environment
env = gym.make("gym_env/DogFight", render_mode = "human")

device = torch.device("cpu")

episodes_done = 0

# Utilize replay memory for more efficient learning (break correlation between samples of experience)
# Use transitions observed by agent, state, action, next state, and resulting reward
# Replay memory also helps "stabilize" the learning process
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Limited replay memory size, sample from past X experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Set up DQN network, layer-by-layer
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, n_actions)

    # Forward pass through NN
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        return self.layer4(x)

# Hyper-parameters for RL training
BATCH_SIZE = 384
GAMMA = 0.99
LR = 1e-4
# Eps-greedy algorithm parameters
EPS_START = 1.00
EPS_END = 0.03
EPS_DECAY = 400
# Update rate of target network
TAU = 0.005

# Get possible # of actions from the environment
n_actions = env.action_space.n
# Reset env to initial state, get initial observations for agent
state, info = env.reset()
# Get the # of observations of the state (size of input layer)
n_observations = len(state)

# Set up policy & target network w/ proper input and output layer sizes
# Learning target constantly shifts as parameters of the DQN are updated.
# This is a problem since it can cause the learning to diverge.
# Separate target network is used to calc. target Q-value.
# Target has same structure as policy NN, but parameters are frozen.
# Target network updated only occasionally to prevent prevent extreme
# divergence or the agent "forgetting" how to act properly.
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
#policy_net.load_state_dict(torch.load("./checkpoints/02499_policy.chkpt"))
#target_net.load_state_dict(torch.load("./checkpoints/02499_target.chkpt"))
#policy_net.eval()
#target_net.eval()

# AdamW optimizer w/ parameters set
optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)
# Set replay memory capacity to first 30 sec of experiences
# over the last 1000 episodes
memory = ReplayMemory(350 * 45 * env.metadata["render_fps"])

# Steps done for eps-greedy algorithm
# As steps grow, make it less likely to choose actions randomly
def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
    if eps_threshold < sample:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

# Track the durations through the episodes of cartpole, high is better (basically track performance for this environment)
episode_durations = []

# Optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # Samples chosen randomly from memory
    transitions = memory.sample(BATCH_SIZE)
    # Get batch of transitions
    # From that acquire states, actions, next states, and rewards
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device = device, dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#    criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 2500
episode_rewards = []
# Train for the desired # of episodes
i = 0
while i < num_episodes:
    ep_step_count = 0
    # Get initial state of episode
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    shooting_transitions = []
    shooting_flags = []
    running_reward = 0
    # Continue until termination
    for t in count():
        # Select action
        action = select_action(state)
        # Get observation, reward, whether we fail or not
        observation, reward, terminated, truncated, info = env.step(action.item())
        running_reward += reward
        ep_step_count += 1
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)

        # Add experience to local memory if it is a shooting state
        # otherwise push to the global memory
        if (
            (info["shoot_id"] is not None) or
            (len(info["hit_ids"]) > 0) or
            (len(info["miss_ids"]) > 0)
        ):
            shooting_transitions.append((state, action, next_state, reward))
            shooting_flags.append(info)
        else:
            memory.push(state, action, next_state, reward)
            optimize_model()
    
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        state = next_state

        if done:
            break

    for x, transition in reversed(list(enumerate(shooting_transitions))):
        for idx, hit_missile_id in reversed(list(enumerate(shooting_flags[x]["hit_ids"]))):
            y = x - 1
            while y >= 0:
                if shooting_flags[y]["shoot_id"] == hit_missile_id:
                    old_reward = shooting_transitions[y][3].item()
                    new_reward = old_reward + shooting_flags[x]["hit_rewards"][idx]
                    running_reward += shooting_flags[x]["hit_rewards"][idx]
                    shooting_transitions[y] = (
                        shooting_transitions[y][0],
                        shooting_transitions[y][1],
                        shooting_transitions[y][2],
                        torch.tensor([new_reward], device = device)
                    )
#                    print(f"Reward for shooting missile {hit_missile_id}: {shooting_flags[x]['hit_rewards'][idx]}")
                y -= 1
        for idx, miss_missile_id in reversed(list(enumerate(shooting_flags[x]["miss_ids"]))):
            y = x - 1
            while y >= 0:
                if shooting_flags[y]["shoot_id"] == miss_missile_id:
                    old_reward = shooting_transitions[y][3].item()
                    new_reward = old_reward + shooting_flags[x]["miss_rewards"][idx]
                    running_reward += shooting_flags[x]["miss_rewards"][idx]
                    shooting_transitions[y] = (
                        shooting_transitions[y][0],
                        shooting_transitions[y][1],
                        shooting_transitions[y][2],
                        torch.tensor([new_reward], device = device)
                    )
                y -= 1
        memory.push(*(transition))
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

    episodes_done += 1
    episode_rewards.append(running_reward)

    if i % 100 == 0 or i == num_episodes - 1:
        torch.save(policy_net.state_dict(), "./checkpoints/{ep:05d}_policy.chkpt".format(ep = i))
        torch.save(target_net.state_dict(), "./checkpoints/{ep:05d}_target.chkpt".format(ep = i))
        with open("memory.pkl", "wb") as file:
            pickle.dump(memory, file)

    print(f"Episode {i:5d} ended, reward: {running_reward}")
    i += 1

print("Complete")
plt.plot(range(num_episodes), episode_rewards)
plt.xlabel("Episode Number")
plt.ylabel("Episode Reward")
plt.title("RL Reward Across Training Episodes")
plt.savefig("./episode_reward_plot.png")
print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")
print(f"Max reward during Episode {episode_rewards.index(max(episode_rewards))}: {max(episode_rewards)}")
with open("replay_buffer.pkl", "wb") as file:
    pickle.dump(memory, file)
