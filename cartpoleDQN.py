# env
import gym

# nn
import torch
import torch.nn as nn

# tools
import random
import matplotlib.pyplot as plt
from tqdm import tqdm



class QNetwork(nn.Module):

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
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        """
        Estimate q-values given state

          Args:
              x (tensor): current state, size (batch x state_size)

          Returns:
              q-values (tensor): estimated q-values, size (batch x action_size)
        """
        return self.net(x)



class DQNAgent:

    def __init__(self, Qnet, targetnet, gamma, lossfn, optim):
        self.Qnet = Qnet
        self.targetnet = targetnet
        self.gamma = gamma
        self.lossfn = lossfn
        self.optim = optim

    def train(self, s0, a0, s1, r, done):
        self.optim.zero_grad()
        # calculate loss
        Qsa = self.Qnet(s0).gather(1, a0.unsqueeze(-1)).squeeze(-1)     # Q(s , a )
        Qsanext = torch.max(self.targetnet(s1), 1)[0]                   # max a' Q(s', a')
        target = r + self.gamma * Qsanext * (1 - done)                  # target = r + gamma * max a' Q(s', a')
        loss = torch.mean(self.lossfn(Qsa, target))                     # loss = (r + gamma * max a' Q(s', a') - Q(s, a)) ^ 2
        # optimization step
        loss.backward()
        self.optim.step()

    def updatetarget(self):
        self.targetnet.load_state_dict(self.Qnet.state_dict())

    def infer(self, x, epsilon, verbose=0):
        with torch.no_grad():
            if random.random() < epsilon:
                return random.randint(0, 1)
            else:
                x = torch.tensor(x).float().unsqueeze(0).to(device)
                Qhat = self.Qnet(x)
                if verbose > 0:
                    print(Qhat)
                return torch.argmax(Qhat).item()


class ReplayBuffer:
    def __init__(self):
        self.memory = []

    def record(self, data):
        self.memory.append(data)

    def prepare_batch(self, batch_size, device):
        """
            Randomly sample batch from memory
            Prepare cuda tensors

            Args:
                memory (list): state, action, next_state, reward, done tuples
                batch_size (int): amount of memory to sample into a batch

            Returns:
                state (tensor): float cuda tensor of size (batch_size x state_size()
                action (tensor): long tensor of size (batch_size)
                next_state (tensor): float cuda tensor of size (batch_size x state_size)
                reward (tensor): float cuda tensor of size (batch_size)
                done (tensor): float cuda tensor of size (batch_size)
        """
        batch_data = random.sample(self.memory, batch_size if batch_size < len(self.memory) else len(self.memory))
        state_ = torch.tensor([x[0] for x in batch_data], dtype=torch.float, device=device)
        action_ = torch.tensor([x[1] for x in batch_data], dtype=torch.long, device=device)
        next_state_ = torch.tensor([x[2] for x in batch_data], dtype=torch.float, device=device)
        reward_ = torch.tensor([x[3] for x in batch_data], dtype=torch.float, device=device)
        done_ = torch.tensor([x[4] for x in batch_data], dtype=torch.float, device=device)
        
        return state_, action_, next_state_, reward_, done_



# Hyper parameters
lr = 1e-3
lossfn = torch.nn.functional.mse_loss
epochs = 500
start_training = 1000
gamma = 0.99
batch_size = 32

epsilon = 1
epsilon_decay = .9999

target_update = 1000
learn_frequency = 2
device = 'cuda'

# Init environment
state_size = 4
action_size = 2
env = gym.make('CartPole-v1', )

# Init networks
q_network = QNetwork(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)

# Init optimizer
optim = torch.optim.Adam(q_network.parameters(), lr=lr)

# Init Learning Agent
learner = DQNAgent(q_network, target_network, gamma, lossfn, optim)
learner.updatetarget()



# Init replay buffer
memory = ReplayBuffer()

# Begin main loop
results_dqn = []
global_step = 0
loop = tqdm(total=epochs, position=0, leave=False)
for epoch in range(epochs):

    # Reset environment
    state = env.reset()
    done = False
    cum_reward = 0  # Track cumulative reward per episode

    # Begin episode
    while not done and cum_reward < 200:  # End after 200 steps 
        # Select e-greedy action
        action = learner.infer(state, epsilon)
        epsilon *= epsilon_decay

        # perform action, receive next observation
        next_state, reward, done, _ = env.step(action)

        # env.render()

        # Store transition (s0, a0, s1, r, done) in replay buffer
        memory.record((state, action, next_state, reward, done))

        # 
        cum_reward += reward
        global_step += 1  # Increment total steps
        state = next_state  # Set current state

        # train for every learn_frequency
        if global_step > start_training and global_step % learn_frequency == 0:

            # Sample batch
            batch = memory.prepare_batch(batch_size, device)
            
            # Train
            learner.train(*batch)

            if global_step % target_update == 0:
                learner.updatetarget()

    # Print results at end of episode
    results_dqn.append(cum_reward)
    loop.update(1)
    loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))



plt.plot(results_dqn)
plt.show()