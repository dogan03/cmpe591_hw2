import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from homework2 import Hw2Env
from model import *
from train import calculate_reward, enhance_state, preprocess_state

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrameCaptureEnv:
    """Wrapper for environment that captures frames for GIF creation"""
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        self.env.reset()
         
        frame = self.env.state()
        if isinstance(frame, torch.Tensor):
             
            frame = frame.permute(1, 2, 0).cpu().numpy()
             
            frame = (frame * 255).astype(np.uint8)
        return frame
    
    def step(self, action):
        obs, reward, terminal, truncated = self.env.step(action)
         
        frame = self.env.state()
        if isinstance(frame, torch.Tensor):
             
            frame = frame.permute(1, 2, 0).cpu().numpy()
             
            frame = (frame * 255).astype(np.uint8)
        return frame, reward, terminal, truncated
    
    def high_level_state(self):
        return self.env.high_level_state()

def save_frames_as_gif(frames, path, scale=2.0, fps=10, quality=85):
    """
    Save a list of frames as a GIF with enhanced quality
    
    Args:
        frames: List of numpy array frames
        path: Path to save the GIF
        scale: Scale factor to resize frames
        fps: Frames per second (controls animation speed)
        quality: Quality for the GIF (higher = better quality but larger file)
    """
     
    duration = int(1000 / fps)
    
     
    valid_frames = [f for f in frames if f is not None]
    
    if not valid_frames:
        print(f"Warning: No valid frames to create GIF at {path}")
        return
    
     
    pil_frames = []
    for frame in valid_frames:
        try:
             
            img = Image.fromarray(frame)
            
             
            width, height = img.size
            
             
            new_width = int(width * scale)
            new_height = int(height * scale)
            
             
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            pil_frames.append(img_resized)
        except Exception as e:
            print(f"Warning: Could not convert frame to PIL image: {e}")
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    if not pil_frames:
        print(f"Warning: No valid PIL frames to create GIF at {path}")
        return
    
     
    try:
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=True,   
            duration=duration,
            loop=0,
            quality=quality   
        )
        print(f"GIF saved to {path} (size: {pil_frames[0].size}, fps: {fps})")
    except Exception as e:
        print(f"Error saving GIF: {e}")
def create_summary_plots(rewards, steps, distances, save_dir):
    """Create and save summary plots"""
    num_episodes = len(rewards)
    
     
    plt.figure(figsize=(15, 5))
    
     
    plt.subplot(131)
    plt.bar(range(1, num_episodes+1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    
     
    plt.subplot(132)
    plt.bar(range(1, num_episodes+1), steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    
     
    plt.subplot(133)
    plt.bar(range(1, num_episodes+1), distances)
    plt.axhline(y=0.03, color='r', linestyle='-', label='Success Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Final Object-Goal Distance')
    plt.title('Final Distance to Goal')
    plt.legend()
    
     
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results_summary.png"))
    print(f"Summary plots saved to {os.path.join(save_dir, 'results_summary.png')}")
    plt.close()

def test_agent(model_path, num_episodes=5, max_steps=300, save_dir='results'):
    """Run test episodes, save GIFs and plot results"""
    
     
    os.makedirs(save_dir, exist_ok=True)
    
     
    N_ACTIONS = 8
    base_env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    env = FrameCaptureEnv(base_env)   
    
     
    base_env.reset()
    raw_state = base_env.high_level_state()
    enhanced_state = enhance_state(raw_state)
    n_states = enhanced_state.shape[0]
    
    print(f"Testing agent from model: {model_path}")
    print(f"Device: {device}")
    
     
    agent = Agent(n_states, N_ACTIONS, HIDDEN_DIM)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.q_local.load_state_dict(checkpoint['model_state_dict'])
            episode_num = checkpoint.get('episode', 0)
            print(f"Loaded checkpoint from episode {episode_num}")
        else:
            agent.q_local.load_state_dict(checkpoint)
            
        agent.q_local.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0
    
     
    success_count = 0
    rewards_all_episodes = []
    steps_all_episodes = []
    goal_distances = []
    
     
    for episode in range(num_episodes):
        print(f"\n--- Running Episode {episode+1}/{num_episodes} ---")
        
         
        initial_frame = env.reset()
        state = env.high_level_state()
        state = preprocess_state(state)
        
         
        frames = [initial_frame]   
        total_reward = 0
        steps = 0
        success = False
        
         
        ee_pos = state[0][:2].cpu().numpy().tolist()
        obj_pos = state[0][2:4].cpu().numpy().tolist()
        goal_pos = state[0][4:6].cpu().numpy().tolist()
        
        print(f"Initial positions - EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}), Object: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}), Goal: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
        
         
        for step in range(max_steps):
             
            with torch.no_grad():
                action = agent.q_local(state).argmax(dim=1).item()
            
             
            frame, reward_env, is_terminal, is_truncated = env.step(action)
            frames.append(frame)   
            
             
            next_state = env.high_level_state()
            next_state = preprocess_state(next_state)
            next_ee_pos = next_state[0][:2].cpu().numpy().tolist()
            next_obj_pos = next_state[0][2:4].cpu().numpy().tolist()
            next_goal_pos = next_state[0][4:6].cpu().numpy().tolist()
            
             
            reward, goal_reached = calculate_reward(
                ee_pos, obj_pos, goal_pos,
                next_ee_pos, next_obj_pos, next_goal_pos
            )
            
             
            total_reward += reward
            steps += 1
            
             
            state = next_state
            ee_pos = next_ee_pos
            obj_pos = next_obj_pos
            goal_pos = next_goal_pos
            
             
            if goal_reached:
                success = True
                print(f"✅ Goal reached in {steps} steps!")
                success_count += 1
                 
                for _ in range(5):
                    frames.append(frame)   
                break
                
            if is_terminal or is_truncated:
                print(f"Episode ended after {steps} steps")
                break
            
             
            time.sleep(0.01)
        
         
        final_obj_goal_dist = np.linalg.norm(np.array(obj_pos) - np.array(goal_pos))
        
         
        rewards_all_episodes.append(total_reward)
        steps_all_episodes.append(steps)
        goal_distances.append(final_obj_goal_dist)
        
         
        print(f"Episode {episode+1} finished:")
        print(f"- Steps: {steps}")
        print(f"- Total reward: {total_reward:.2f}")
        print(f"- Final object-goal distance: {final_obj_goal_dist:.3f}")
        print(f"- Success: {'Yes' if success else 'No'}")
        
         
        gif_path = os.path.join(save_dir, f"episode_{episode+1}.gif")
        save_frames_as_gif(frames, gif_path)
    
     
    print("\n--- Test Summary ---")
    print(f"Success rate: {success_count/num_episodes:.2f} ({success_count}/{num_episodes})")
    print(f"Average reward: {np.mean(rewards_all_episodes):.2f}")
    print(f"Average steps: {np.mean(steps_all_episodes):.2f}")
    print(f"Average final distance to goal: {np.mean(goal_distances):.3f}")
    
     
    create_summary_plots(
        rewards_all_episodes, 
        steps_all_episodes, 
        goal_distances, 
        save_dir
    )
    
    return success_count/num_episodes

if __name__ == "__main__":
    import argparse

     
    random_seed = 1
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
     
    parser = argparse.ArgumentParser(description='Test DQN agent with GIF output')
    parser.add_argument('--model', type=str, default="hw2_model.pth", 
                        help='Path to the model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to test')
    parser.add_argument('--steps', type=int, default=300, 
                        help='Maximum steps per episode')
    parser.add_argument('--save-dir', type=str, default='test_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
     
    test_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        save_dir=args.save_dir
    )