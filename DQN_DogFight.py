import pickle
import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import pandas as pd

import gym_env_AlexNet

from collections import deque, namedtuple
from itertools import count
from tensordict import TensorDict
from torch import nn, optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
import torch

torch.cuda.empty_cache()
#This has been added in order to empty the cache



save_image = False
#Save Image is a flag added in order to determine whether the program needs to save the images before inserting into the Convolutional Neural Network. 

Running_Mode = False
#Running_Mode is a flag added in order to detemine whether the program is finished training and ready to be evaluated. In running mode, we have the weights of the neural network's 9999th checkpoint extracted. We also set the network to always choose the optimal action. We also changed the view to human mode so you can see what the agent is doing. 
#Usually we set Running mode to be false when we are training the RL algorithm from scratch. This is because there is no checkpoint evaluated. Human rending is off because it would be inefficient to see the agent on screen while training. 

# Custom wrappers

class SkipFrame(gym.Wrapper):
#Skipframe(gym.Wrapper) class	
    def __init__(self, env, skip):
	    #Intialize envrionment 
        """Return only every `skip`-th frame"""
        super().__init__(env)
	#Try to connect to a higher class    
        self._skip = skip
	#Skip the appropriate frames    

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            #lets look at the obs, reward, done, trunk and info
            total_reward += reward
            #We take the total
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
	#we are trying to combine it with the ObservationWrapper class    
        obs_shape = self.observation_space.shape[:2]
	#We are taking the observative_space.shape[:2]    
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
	#We are putting everything into gym.spaces.box()
    def permute_orientation(self, observation):
	
        observation = np.transpose(observation, (2, 0, 1))
	#We are taking an observation vector and transposing it by (2,0,1)    
        observation = torch.tensor(observation.copy(), dtype=torch.float)
	# We are taking an observation vector copying it and putting in a torch.tensor
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
	#we are permuting the orientation of the observation
        transform = T.Grayscale()
	#We are transforming the T vector and applying Grayscale onto it
        observation = transform(observation)
	#We are taking the observation then transforming it. 
        return observation


class ResizeObservation(gym.ObservationWrapper):
	#This class is just used to resize the vector
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
render__mode = None
if Running_Mode:
	render__mode = "human"

env = gym.make("gym_env_AlexNet/DogFight", render_mode = render__mode)
#We are making the AlexNet DogFight code 
env = SkipFrame(env, skip = 1)
# We are skipping a single frame
env = GrayScaleObservation(env)
#We are converting the observation space into GrayScale 
env = ResizeObservation(env, shape = 400)
#We are resizing the observation shape
env = gym.wrappers.FrameStack(env, num_stack = 4, lz4_compress = False)
#We are applying FrameStack to the env, num_stack and lz4_compress



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Use cuda.is_available() else use the 'cpu'


data_min = 0
#Take the minimium data
data_max = 255
#Take the maximium data point


def normalize(data):
    return (data-data_min)/(data_max-data_min)
    #We are normalizing the data by substracting the minimium and then dividing by the range of the data. 
    
    
def imshow(img):
    npimg = img.cpu().numpy()
    #Img.cpu().numpy()
    return npimg
    
    







episodes_done = 0

# Set up DQN network, layer-by-layer
#This specific CNN is AlexNet which is a convolutional neural network that is widely used. AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense layers as the basic building blocks. Grouped convolutions are used in order to fit the model across two GPUs.

class DQN(nn.Module):
    def __init__(self, n_state_dim, n_actions):
        super().__init__()
        c, w, h = n_state_dim
	#C,W,H is state_dimmesnions     
        self.features = nn.Sequential(
	    	
            nn.Conv2d(c, 64, kernel_size=11, stride=4, padding=2),
	    #We are taking the conv2d vector with c channel, 64, kernel_size = 11, stride = 4, and padding is 2
            nn.ReLU(inplace=True),
	    #The ReLU will just take the output of the conv2d and apply the ReLU filter to it. 
            nn.MaxPool2d(kernel_size=3, stride=2),
	    #We will take the output of the ReLU and apply a MaxPool2d filter 	
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
	    #We will take the output of the MaxPool2d filter and apply a 2d convolution	
            nn.ReLU(inplace=True),
	    #We then apply another RELU filter 
            nn.MaxPool2d(kernel_size=3, stride=2),
	    #We then take Maxpool2d keneral size of 3 and stride of 2	
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
	    #We then apply a 2d convolution by applying a kernel size of 3 and a padding of 1
            nn.ReLU(inplace=True),
	    #We then apply another RELU filter 	
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
	    #We then apply a 2d convolution	
            nn.ReLU(inplace=True),
	    #We then apply ReLU	on a convolutional 2d filter 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
	    #We then apply a 2d convolutio
            nn.ReLU(inplace=True),
	    #We then apply a ReLU filter
            nn.MaxPool2d(kernel_size=3, stride=2),
	    #We then apply a MaxPool2d filter  
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
	#Self.avgpool is the adaptive Avg Pool2D (6,6)
	    
        self.classifier = nn.Sequential(
	    #This creates a Sequential Neural Network	
            nn.Dropout(),
	    #This creates a Dropout 
            nn.Linear(256 * 6 * 6, 4096),
	    #Linear layer from standard AlexNet structure
            nn.ReLU(inplace=True),
	    #This is a ReLU filter	
            nn.Dropout(),
	    #This is a dropout	
            nn.Linear(4096, 4096),
	    #This is a Linear Layer	
            nn.ReLU(inplace=True),
	    #The is a ReLU function
            nn.Linear(4096, n_actions),
	    #This is a linear layer with 4096 transactions
        )

    def _forward_conv(self, x):
	#We apply each layer of a convolution function and relu activation function   
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
	#Take the x vector     
        x = self.features(x)
	#Take the x vector and feed it into the features function
        x = self.avgpool(x)
	#Take the x vector and feed it to the average pooling function    
        x = torch.flatten(x, 1)
	#Take the x vector and feed it to the flatten torch function    
        x = self.classifier(x)
	#Take the x vector and feed it into the classifier function
        return x


# Hyper-parameters for RL training
BATCH_SIZE = 32
GAMMA = 0.99
LR = 2.5e-4
# Eps-greedy algorithm parameters
EPS_START = 1.00
EPS_END = 0.1
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

policy_net = DQN(state.shape, n_actions).to(device)
target_net = DQN(state.shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
if Running_Mode:
	policy_net.load_state_dict(torch.load("./checkpoints/09999_policy.chkpt"))
	target_net.load_state_dict(torch.load("./checkpoints/09999_policy.chkpt"))
	policy_net.eval()
	target_net.eval()

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

# Steps done for eps-greedy algorithm
# As steps grow, make it less likely to choose actions randomly
def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
    if Running_Mode:
    	eps_threshold = 0
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
    # Samples chosen randomly from memory
    batch = memory.sample().to(device)
    states, actions, next_states, rewards, terminations = (batch.get(key) for key in ("state", "action", "next_state", "reward", "terminated"))
    actions = actions.squeeze()
    rewards = rewards.squeeze()
    terminations = terminations.squeeze()
  
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1))
    
    

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
    imnum = 0
    for I in imshow(torch.from_numpy(np.array(state))):
	#This for loop takes the state (which in the envvrionment is defined by an image of the entire gym environment) and creates a image out of the state. since each state has 4 images. We essentially loop through all four images and create an image of out the state.    
    	imnum += 1
    	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    	img = Image.fromarray(I8)
    	if save_image:
	#When save_image flag is turned on this basically takes the image then saves it onto Episode with i representing the episode number. inum is the # in the sequence of 4 images
    		img.save(f"./images/Episode:{i}:0:{imnum}.png")

    state = torch.from_numpy(normalize(np.array(state))).to(device)
    #We are taking the numpy array and then normalizing the state array to make the Neural Network calculations easier. We normalize it after saving the image to keep the image quality in tact. Keep in mind this step is bewteen the image being saved and the state being fed into the select action function (this function essentially feeds it into the neural network)

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
            
        imnum = 0    
        
        for I in imshow(torch.from_numpy(np.array(next_state))):
		#The for loop creates the image from the next_state vector.
        	imnum += 1
        	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        	img = Image.fromarray(I8)
        	if save_image:
		#When save_image flag is turned on this basically takes the image then saves it onto Episode with i representing the episode number. inum is the # in the sequence of 4 images
        		img.save(f"./images/Episode:{i}:{t}:{imnum}.png")
        next_state = torch.from_numpy(normalize(np.array(next_state))).to(device)
	#Here we normalize the next_state vector to make the Neural Network calculations easier. We normalize it after saving the image to keep the image quality in tact.    
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
