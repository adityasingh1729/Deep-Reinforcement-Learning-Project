import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from PIL import Image
import matplotlib.pyplot as plt

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Preprocessing for the Pong environment
def preprocess(image):
    if isinstance(image, tuple):
        image = image[0]  # Extract the image from the tuple

    # Convert NumPy array to PIL image
    image = Image.fromarray(image)

    resize_transform = Compose([Resize((80, 80)), Grayscale(), ToTensor()])
    return resize_transform(image).view(1, -1).float()

# Experience Replay buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Double Deep Q-Network (DDQN) Agent
class DDQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, batch_size=64, lr=1e-4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.lr = lr

        self.input_dim = 80 * 80  # Size after preprocessing
        self.output_dim = self.env.action_space.n

        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(capacity=10000)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes=1000):
        episode_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            state = preprocess(state)
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                step_result = self.env.step(action)
                if len(step_result) < 4:
                    raise ValueError("Unexpected step_result format")
                next_state, reward, done, _ = step_result[:4]  # Adjust the unpacking based on step_result length
                next_state = preprocess(next_state)
                total_reward += reward

                self.memory.push((state, action, reward, next_state, done))
                state = next_state

                if len(self.memory.memory) >= self.batch_size:
                    self.optimize_model()

            self.update_target_network()
            self.update_epsilon()

            episode_rewards.append(total_reward)
            print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {self.epsilon}")

        return episode_rewards

    def optimize_model(self):
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * ~dones

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Initialize the environment and agent
env = gym.make("PongNoFrameskip-v4")
agent = DDQNAgent(env)

# Train the agent and get episode rewards
episode_rewards = agent.train(num_episodes=1000)

# Plotting the average rewards per episode
def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Average Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rewards(episode_rewards)

# Save the trained model
torch.save(agent.policy_net.state_dict(), 'pong_ddqn_model.pth')
