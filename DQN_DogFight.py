import pickle
import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from typing import Type
import gym_env_ResNet
from torch import nn, optim

from collections import deque, namedtuple
from itertools import count
from tensordict import TensorDict
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn

# Custom wrappers

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):

        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias = None), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Define gym environment and apply wrappers
env = gym.make("gym_env_ResNet/DogFight")
env = SkipFrame(env, skip = 1)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape = 100)
env = gym.wrappers.FrameStack(env, num_stack = 4, lz4_compress = False)

#env = gym.wrappers.FrameStack(env, 4, lz4_compress = True)

device = torch.device( 'cpu')

episodes_done = 0

# Set up DQN network, layer-by-layer

class BasicBlock(nn.Module):
    """
    Builds the Basic Block of the ResNet model.
    For ResNet18 and ResNet34, these are stackings od 3x3=>3x3 convolutional
    layers.
    For ResNet50 and above, these are stackings of 1x1=>3x3=>1x1 (BottleNeck) 
    layers.
    """
    def __init__(
        self, 
        num_layers: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.num_layers = num_layers
        # Multiplicative factor for the subsequent conv2d layer's output 
        # channels.
        # It is 1 for ResNet18 and ResNet34, and 4 for the others.
        self.expansion = expansion
        self.downsample = downsample
        # 1x1 convolution for ResNet50 and above.
        if num_layers > 34:
            self.conv0 = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=1,
                bias=False
            )
            self.bn0 = nn.BatchNorm2d(out_channels)
            in_channels = out_channels
        # Common 3x3 convolution for all.
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 1x1 convolution for ResNet50 and above.
        if num_layers > 34:
            self.conv2 = nn.Conv2d(
                out_channels, 
                out_channels*self.expansion, 
                kernel_size=1, 
                stride=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        else:
            # 3x3 convolution for ResNet18 and ResNet34 and above.
            self.conv2 = nn.Conv2d(
                out_channels, 
                out_channels*self.expansion, 
                kernel_size=3, 
                padding=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
    
        # Through 1x1 convolution if ResNet50 or above.
        if self.num_layers > 34:
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.relu(out)
        # Use the above output if ResNet50 and above.
        if self.num_layers > 34:
            out = self.conv1(out)
        # Else use the input to the `forward` method.
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out


class DQN(nn.Module):
    def __init__(self, n_state_dim,num_layers, block, n_actions):
        super().__init__()
        c, w, h = n_state_dim
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        if num_layers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1
        if num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        if num_layers == 101:
            layers = [3, 4, 23, 3]
            self.expansion = 4
        if num_layers == 152:
            layers = [3, 8, 36, 3]
            self.expansion = 4
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_layers=num_layers)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_layers=num_layers)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_layers=num_layers)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, n_actions)


    def _make_layer(self, block: Type[BasicBlock],out_channels: int,blocks: int,stride: int = 1,num_layers: int = 18):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                num_layers, 
                self.in_channels, 
                out_channels, 
                stride, 
                self.expansion, 
                downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                num_layers,
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Hyper-parameters for RL training
BATCH_SIZE = 32
GAMMA = 0.99
LR = 2.5e-4
# Eps-greedy algorithm parameters
EPS_START = 1.00
EPS_END = 0.3
EPS_DECAY = 3000
# Update rate of target network
TAU = 0.0025

# Get possible # of actions from the environment
n_actions = env.action_space.n
# Reset env to initial state, get initial observations for agent
state, info = env.reset()
print(f"State shape: {state.shape}")
# Get the # of observations of the state (size of input layer)
n_observations = len(state)

# Set up policy & target network w/ proper input and output layer sizes
# Learning target constantly shifts as parameters of the DQN are updated.
# This is a problem since it can cause the learning to diverge.
# Separate target network is used to calc. target Q-value.
# Target has same structure as policy NN, but parameters are frozen.
# Target network updated only occasionally to prevent prevent extreme
# divergence or the agent "forgetting" how to act properly.
#policy_net = DQN(n_observations, n_actions).to(device)
#target_net = DQN(n_observations, n_actions).to(device)
policy_net = DQN(state.shape, 18, BasicBlock, n_actions).to(device)
target_net = DQN(state.shape, 18, BasicBlock, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
for p in target_net.parameters():
    p.requires_grad = False
#policy_net.load_state_dict(torch.load("./checkpoints3/09999_policy.chkpt"))
#target_net.load_state_dict(torch.load("./checkpoints3/09999_policy.chkpt"))
#policy_net.eval()
#target_net.eval()

# AdamW optimizer w/ parameters set
optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)
memory = TensorDictPrioritizedReplayBuffer(
    alpha = 0.65,
    beta = 0.45,
    eps = 1e-6,
    storage = LazyMemmapStorage(
        max_size = 100000,
        device = device
    ),
    batch_size = BATCH_SIZE,
    pin_memory = False
)
#memory.load_state_dict(torch.load("./checkpoints/00999_memory.chkpt"))

# Steps done for eps-greedy algorithm
# As steps grow, make it less likely to choose actions randomly
def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
    if eps_threshold < sample:
        with torch.no_grad():
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)

    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

# Track the durations through the episodes of cartpole, high is better (basically track performance for this environment)
episode_durations = []

# Optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample().to(device)
    states, actions, next_states, rewards, terminations = (batch.get(key) for key in ("state", "action", "next_state", "reward", "terminated"))
    actions = actions.squeeze()
    rewards = rewards.squeeze()
    terminations = terminations.squeeze()
    #print(f"Shapes: {states.shape}, {actions.shape}, {next_states.shape}, {rewards.shape}, {terminations.shape}")
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1))
#    state_action_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]

    expected_state_action_values = rewards + (1. - terminations.float()) * GAMMA * next_state_values

    criterion = nn.MSELoss(reduction = "none")
    td_errors = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    weights = batch.get("_weight")
    loss = (weights * td_errors).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()

    batch.set("td_error", td_errors)
    memory.update_tensordict_priority(batch)

num_episodes = 10000
episode_rewards = []
# Train for the desired # of episodes
i = 0
for i in range(num_episodes):
    ep_step_count = 0
    # Get initial state of episode
    state, info = env.reset()
#    state = np.array(state)
    state = torch.from_numpy(np.array(state)).to(device)
#    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    running_reward = 0
    # Continue until termination
    for t in count():
        # Select action
        action = select_action(state)
        # Get observation, reward, whether we fail or not
        next_state, reward, terminated, truncated, info = env.step(action.item())
        running_reward += reward
        ep_step_count += 1
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated

        if done:
            break

        next_state = torch.from_numpy(np.array(next_state)).to(device)
        state = next_state

        terminated = torch.tensor([terminated], dtype = torch.bool, device = device)
        memory.add(TensorDict({"state": state, "action": action, "next_state": next_state, "reward": reward, "terminated": terminated}, batch_size = []))
        optimize_model()
    
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

        state = next_state

    episodes_done += 1
    episode_rewards.append(running_reward)
    if (i + 1) % 100 == 0 or (i + 1) == num_episodes:
        torch.save(policy_net.state_dict(), f"./checkpoints/{i:05d}_policy.chkpt")
        torch.save(target_net.state_dict(), f"./checkpoints/{i:05d}_target.chkpt")
        #torch.save(memory.state_dict(), f"./checkpoints/{i:05d}_memory.chkpt")

    print(f"Episode {i:5d} ended, reward: {running_reward}")

s = pd.Series(episode_rewards)
s_ma = s.rolling(10).mean()
print("Complete")
fig, ax = plt.subplots()
ax.plot(s, label = "Raw Rewards")
ax.plot(s_ma, label = "Rewards (moving average)")
ax.legend()
plt.xlabel("Episode Number")
plt.ylabel("Episode Reward")
plt.title("RL Reward Across Training Episodes")
plt.savefig("./episode_reward_plot.png")
print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")
print(f"Max reward during Episode {episode_rewards.index(max(episode_rewards))}: {max(episode_rewards)}")
