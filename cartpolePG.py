# cartpole using policy gradient
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Policy(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 8
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)



import random

class PGAgent:

    def __init__(self, Pnet, optim):
        self.Pnet = Pnet
        self.optim = optim
        self.Pnet.to(device)

    def train(self, obs, actions, reward, b=0):
        """ optimize using policy gradient """
        self.optim.zero_grad()
        probs = torch.squeeze(torch.gather(self.Pnet(obs), 1, torch.unsqueeze(actions.long(), dim=1)))
        # J = - E[ sum(log(pi(a|s)) * reward )]
        J = - torch.sum(torch.log(probs) * reward)
        J.backward()
        self.optim.step()
        return J.item()

    def infer(self, x, epsilon):
        with torch.no_grad():
            if random.random() < epsilon:
                return random.randint(0, 1)
            else:
                x = torch.tensor(x).float().unsqueeze(0).to(device)
                phat = self.Pnet(x)
                return torch.argmax(phat).item()



# init environment
import gym
env = gym.make('CartPole-v1', )

# init agent
lr = 1e-3
policy = Policy(4, 2)
optim  = torch.optim.Adam(policy.parameters(), lr=lr)
agent  = PGAgent(policy, optim)

# agent
epsilon = 1
epsilon_decay = 0.99999
gamma = 0.99

# training
epochs = 3000
batch_size = 5
# iteration = 20



# init history
observations = []
rewards      = []
actions      = []
# training loop
total_rwds = []
epoch=0
while epoch < 2300:
    state = env.reset()
    done = False
    cumulative_reward = 0
    expected_returns = []
    
    # Rollout
    while not done:
        
        # e-greedy action selection
        action = agent.infer(state, epsilon)
        actions.append(action)
        epsilon *= epsilon_decay
        
        # perform action, receive next observation
        next_state, reward, done, _ = env.step(action)
        if done: reward = 0
        cumulative_reward += reward
        
        # no render to speedup training
        # env.render()
        
        # store in history
        observations.append(state)
        expected_returns.append(reward)
        state = next_state
    
    # calculate discounted total expected return at t
    for i in range(-2, -len(expected_returns)-1, -1):
        expected_returns[i] = expected_returns[i+1] + expected_returns[i] * gamma
    rewards.extend(expected_returns)

    # on policy training
    
    # prepare data
    obs = torch.tensor(observations, dtype=torch.float32).to(device)
    rwd = torch.tensor(rewards, dtype=torch.float32).to(device)
    act = torch.tensor(actions, dtype=torch.float32).to(device)
    
    # clear history
    observations.clear()
    rewards.clear()
    actions.clear()
    
    # update policy
    J = agent.train(obs, act, rwd)
    
    # record reward
    total_rwds.append(cumulative_reward)
    print("Episode:", epoch, "return:", cumulative_reward)
    epoch+=1