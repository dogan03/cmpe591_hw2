import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from homework2 import Hw2Env
from model import DQNAgent

writer = SummaryWriter()
N_ACTIONS = 8
agent = DQNAgent(state_shape=(3, 128, 128), n_actions=N_ACTIONS)
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
N_EPISODES = 100
for episode in range(N_EPISODES):
    env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    # random state
    state, _, _, _ = env.step(np.random.randint(N_ACTIONS))
    episode_loss = 0
    while not done:
        action = agent.act(state)
        next_state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        state = next_state
        total_reward += reward
        episode_steps += 1
        if loss is not None:
            episode_loss += loss


    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("RPS", total_reward/episode_steps, episode)
    writer.add_scalar("Epsilon", agent.epsilon, episode)
    writer.add_scalar("Average Loss", episode_loss/episode_steps, episode)
    writer.flush()
    print(f"Episode={episode}, reward={total_reward}, RPS={total_reward/episode_steps}")