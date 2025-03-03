import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from homework2 import Hw2Env
from model import *

writer = SummaryWriter()

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enhance_state(high_level_state):
    """
    Enhance the raw state with additional features to improve learning
    
    Args:
        high_level_state: Raw state from environment (ee_pos, obj_pos, goal_pos)
        
    Returns:
        Enhanced state with additional features
    """
     
    if isinstance(high_level_state, torch.Tensor):
        high_level_state = high_level_state.cpu().numpy()
    
    ee_pos = high_level_state[:2]
    obj_pos = high_level_state[2:4]
    goal_pos = high_level_state[4:6]
    
     
    ee_to_obj = obj_pos - ee_pos   
    obj_to_goal = goal_pos - obj_pos   
    ee_to_goal = goal_pos - ee_pos   
    
     
    ee_obj_dist = np.linalg.norm(ee_to_obj)
    obj_goal_dist = np.linalg.norm(obj_to_goal)
    ee_goal_dist = np.linalg.norm(ee_to_goal)
    
     
     
    if ee_obj_dist > 0 and obj_goal_dist > 0:
        angle_ee_obj_goal = np.arccos(
            np.clip(np.dot(ee_to_obj, obj_to_goal) / (ee_obj_dist * obj_goal_dist), -1.0, 1.0)
        )
    else:
        angle_ee_obj_goal = 0
        
     
     
    ideal_push_vector = -obj_to_goal / (obj_goal_dist if obj_goal_dist > 0 else 1)
    actual_approach_vector = ee_to_obj / (ee_obj_dist if ee_obj_dist > 0 else 1)
    push_angle_alignment = np.dot(ideal_push_vector, actual_approach_vector)
    
     
    enhanced_state = np.concatenate([
        ee_pos,                   
        obj_pos,                  
        goal_pos,                 
        ee_to_obj,                
        obj_to_goal,              
        [ee_obj_dist],            
        [obj_goal_dist],          
        [angle_ee_obj_goal],      
        [push_angle_alignment],   
    ])
    
    return enhanced_state

 
def preprocess_state(state):
    """Convert state to appropriate tensor format with enhanced features"""
    if state is None:
        return None
        
     
    if isinstance(state, np.ndarray) and state.size >= 6:
        enhanced = enhance_state(state)
        return torch.from_numpy(enhanced.astype(np.float32)).float().unsqueeze(0).to(device)
    elif isinstance(state, torch.Tensor) and state.numel() >= 6:
         
        if state.dim() > 1 and state.shape[0] == 1:
             
            state_np = state.squeeze(0).cpu().numpy()
        else:
            state_np = state.cpu().numpy()
            
        enhanced = enhance_state(state_np)
        return torch.from_numpy(enhanced).float().unsqueeze(0).to(device)
    else:
         
        if isinstance(state, np.ndarray):
            return torch.from_numpy(state.astype(np.float32)).float().unsqueeze(0).to(device)
        elif isinstance(state, torch.Tensor):
            return state.float().unsqueeze(0).to(device) if state.dim() == 1 else state.float().to(device)
        else:
            return torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(device)
def calculate_reward(prev_ee_pos, prev_obj_pos, prev_goal_pos, 
                    curr_ee_pos, curr_obj_pos, curr_goal_pos):
    """
    Calculate reward based on the agent's actions and resulting state
    
    Focused on pushing the object toward the goal
    """
     
    reward = 0
    
     
    prev_ee_obj_dist = np.linalg.norm(np.array(prev_ee_pos) - np.array(prev_obj_pos))
    curr_ee_obj_dist = np.linalg.norm(np.array(curr_ee_pos) - np.array(curr_obj_pos))
    
    prev_obj_goal_dist = np.linalg.norm(np.array(prev_obj_pos) - np.array(prev_goal_pos))
    curr_obj_goal_dist = np.linalg.norm(np.array(curr_obj_pos) - np.array(curr_goal_pos))
    
     
    ee_obj_approaching = prev_ee_obj_dist - curr_ee_obj_dist
    reward += ee_obj_approaching * 2.0   
    
     
    if curr_ee_obj_dist < 0.05:
        reward += 1.0
    
     
    obj_goal_improvement = prev_obj_goal_dist - curr_obj_goal_dist
    reward += obj_goal_improvement * 5.0   
    
     
    if curr_obj_goal_dist < 0.1:
        reward += 2.0
    if curr_obj_goal_dist < 0.05:
        reward += 5.0
    
     
    obj_moved = not (abs(prev_obj_pos[0] - curr_obj_pos[0]) < 0.001 and 
                   abs(prev_obj_pos[1] - curr_obj_pos[1]) < 0.001)
    if not obj_moved:
        reward -= 0.1
    
     
    goal_reached = curr_obj_goal_dist < 0.03
    if goal_reached:
        reward += 10.0
    
    return reward, goal_reached

def train_agent(num_episodes=2000, max_steps=200, render_mode="offscreen", checkpoint_path=None):
    """
    Train agent to push an object to a goal position
    
    Simplified approach with clear objective and dense rewards
    
    Args:
        num_episodes: Maximum number of episodes to train for
        max_steps: Maximum steps per episode
        render_mode: Rendering mode for environment
        checkpoint_path: Optional path to load a checkpoint from
    """
     
    N_ACTIONS = 8
    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)
    
     
    env.reset()
    raw_state = env.high_level_state()
    enhanced_state = enhance_state(raw_state)
    n_states = enhanced_state.shape[0]   
    
    print(f"Raw state dimension: {len(raw_state)}")
    print(f"Enhanced state dimension: {n_states}, Action dimension: {N_ACTIONS}")
    
     
    agent = Agent(n_states, N_ACTIONS, HIDDEN_DIM)
    
     
    start_episode = 0
    scores = []
    recent_scores = deque(maxlen=100)
    success_count = 0
    success_rate_history = []
    epsilon = EPSILON_START
    
     
    if checkpoint_path:
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
             
            agent.q_local.load_state_dict(checkpoint['model_state_dict'])
            agent.q_target.load_state_dict(checkpoint['target_state_dict'])
            agent.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            
             
            start_episode = checkpoint['episode'] -1
            scores = checkpoint['scores'] if 'scores' in checkpoint else []
            success_rate_history = checkpoint['success_rate_history'] if 'success_rate_history' in checkpoint else []
            
             
            if scores:
                for s in scores[-100:]:
                    recent_scores.append(s)
            
             
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** start_episode))
            
            print(f"Loaded checkpoint from episode {start_episode}")
            print(f"Current epsilon: {epsilon:.4f}")
            if 'last_success_rate' in checkpoint and checkpoint['last_success_rate'] is not None:
                print(f"Previous success rate: {checkpoint['last_success_rate']:.2f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training...")
    
     
    best_avg_score = np.mean(recent_scores) if recent_scores else -float('inf')
    patience = float('inf')   
    patience_counter = 0
    
     
    os.makedirs("models", exist_ok=True)
    
    try:
        print(f"Starting training from episode {start_episode+1}...")
        for episode in range(start_episode, num_episodes):
             
            env.reset()
            state = env.high_level_state()
            state = preprocess_state(state)
            
             
            ee_pos = state[0][:2].tolist()
            obj_pos = state[0][2:4].tolist() 
            goal_pos = state[0][4:6].tolist()
            
             
            score = 0
            episode_steps = 0
            success = False
            
             
            for step in range(max_steps):
                 
                action = agent.get_action(state, epsilon)
                next_state_raw, reward_env, is_terminal, is_truncated = env.step(action.item())
                
                 
                next_state = env.high_level_state()
                next_state = preprocess_state(next_state)
                next_ee_pos = next_state[0][:2].tolist()
                next_obj_pos = next_state[0][2:4].tolist()
                next_goal_pos = next_state[0][4:6].tolist()
                
                 
                reward, goal_reached = calculate_reward(
                    ee_pos, obj_pos, goal_pos,
                    next_ee_pos, next_obj_pos, next_goal_pos
                )
                
                 
                done = False
                if is_terminal or is_truncated or goal_reached:
                    if goal_reached:
                        success = True
                        reward += 20.0   
                        print(f"Episode {episode+1}: Goal reached in {step+1} steps!")
                    done = True
                
                 
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
                done_tensor = torch.tensor([float(done)], dtype=torch.float32, device=device)
                agent.step(state, action, reward_tensor, next_state, done_tensor)
                
                 
                state = next_state
                ee_pos = next_ee_pos
                obj_pos = next_obj_pos
                goal_pos = next_goal_pos
                score += reward
                episode_steps += 1
                
                 
                if done:
                    break
            
             
            if success:
                success_count += 1
                
            success_rate = None
            if (episode + 1) % 100 == 0:
                success_rate = success_count / 100
                success_rate_history.append(success_rate)
                print(f"Success rate over last 100 episodes: {success_rate:.2f}")
                success_count = 0   
            
             
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
             
            scores.append(score)
            recent_scores.append(score)
            avg_score = np.mean(recent_scores)
            
             
            print(f"Episode {episode+1}/{num_episodes}, Score: {score:.2f}, Avg(100): {avg_score:.2f}, Steps: {episode_steps}, Epsilon: {epsilon:.3f}")
            
             
            writer.add_scalar("Training/Score", score, episode)
            writer.add_scalar("Training/AvgScore", avg_score, episode)
            writer.add_scalar("Training/Steps", episode_steps, episode)
            writer.add_scalar("Training/Epsilon", epsilon, episode)
            writer.add_scalar("Training/Success", float(success), episode)
            
             
            if (episode + 1) % 1000 == 0:
                checkpoint = {
                    'episode': episode + 1,
                    'model_state_dict': agent.q_local.state_dict(),
                    'target_state_dict': agent.q_target.state_dict(),
                    'optimizer_state_dict': agent.optim.state_dict(),
                    'scores': scores,
                    'success_rate_history': success_rate_history,
                    'last_success_rate': success_rate
                }
                torch.save(checkpoint, f"models/dqn_checkpoint_ep{episode+1}.pth")
                print(f"Checkpoint saved at episode {episode+1}")
            
             
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                patience_counter = 0
                 
                checkpoint = {
                    'episode': episode + 1,
                    'model_state_dict': agent.q_local.state_dict(),
                    'target_state_dict': agent.q_target.state_dict(),
                    'optimizer_state_dict': agent.optim.state_dict(),
                    'scores': scores,
                    'success_rate_history': success_rate_history,
                    'last_success_rate': success_rate
                }
                torch.save(checkpoint, "models/dqn_agent_best.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= patience and episode > 500:   
                print(f"Early stopping after {episode+1} episodes due to no improvement")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        checkpoint = {
            'episode': episode + 1,
            'model_state_dict': agent.q_local.state_dict(),
            'target_state_dict': agent.q_target.state_dict(),
            'optimizer_state_dict': agent.optim.state_dict(),
            'scores': scores,
            'success_rate_history': success_rate_history,
            'last_success_rate': None
        }
        torch.save(checkpoint, "models/dqn_checkpoint_interrupted.pth")
        print("Checkpoint saved. You can resume training using this checkpoint.")
    
     
    checkpoint = {
        'episode': num_episodes if start_episode == num_episodes else episode + 1,
        'model_state_dict': agent.q_local.state_dict(),
        'target_state_dict': agent.q_target.state_dict(),
        'optimizer_state_dict': agent.optim.state_dict(),
        'scores': scores,
        'success_rate_history': success_rate_history,
        'last_success_rate': success_rate_history[-1] if success_rate_history else None
    }
    torch.save(checkpoint, "dqn_agent_final.pth")
    
     
    if success_rate_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(100, start_episode + len(success_rate_history)*100 + 1, 100), success_rate_history)
        plt.title("Success Rate History (per 100 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.grid(True)
        plt.savefig("success_rate.png")
    
     
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title("DQN Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig("training_progress.png")
    
    print("\nTraining Complete!")
    return agent, scores

if __name__ == "__main__":
     
    import os
    import sys
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print("Device:", device)
    
     
    checkpoint_path = "models/dqn_checkpoint_ep5000_2.pth"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"Will resume training from checkpoint: {checkpoint_path}")
        
     
    env_debug = Hw2Env(n_actions=8, render_mode="offscreen")
    env_debug.reset()
    raw_state = env_debug.high_level_state()
    enhanced = enhance_state(raw_state)
    print(f"Raw state: shape={np.array(raw_state).shape}, values={raw_state}")
    print(f"Enhanced state: shape={enhanced.shape}, first few values={enhanced[:5]}")
    
     
    agent, scores = train_agent(
        num_episodes=1000000,   
        max_steps=300,       
        render_mode="offscreen", 
        checkpoint_path=checkpoint_path
    )