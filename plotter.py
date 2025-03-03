import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

 
 
df = pd.read_csv('score.csv')

 
episodes = df['Step']   
scores = df['Value']    

 
plt.figure(figsize=(12, 6))

 
plt.scatter(episodes, scores, color='lightblue', s=10, alpha=0.4, label='Individual Episode Scores')

 
 
window_size = min(51, len(scores) - 1 if len(scores) % 2 == 0 else len(scores))   
if window_size > 3:   
    poly_order = 3     
    try:
        smoothed_scores = savgol_filter(scores, window_size, poly_order)
        plt.plot(episodes, smoothed_scores, '-', color='blue', linewidth=2, 
                label=f'Smoothed Trend (window={window_size})')
    except:
        print("Could not apply Savgol filter - insufficient data or invalid parameters")

 
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

ma_window = min(30, len(scores) // 3)   
if ma_window >= 2:   
    ma_smoothed = moving_average(scores, ma_window)
     
    ma_episodes = episodes[ma_window-1:]
    plt.plot(ma_episodes, ma_smoothed, '-', color='red', linewidth=2, 
            label=f'Moving Average (window={ma_window})')

 
if len(scores) > 0:
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    plt.axhline(y=avg_score, color='green', linestyle='--', 
               label=f'Average Score: {avg_score:.2f}')

     
    max_score_episode = episodes[np.argmax(scores)]
    plt.scatter(max_score_episode, max_score, color='gold', s=100, zorder=5, 
               label=f'Max Score: {max_score:.2f}')

 
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Training Score Progression', fontsize=16)

 
plt.grid(True, linestyle='--', alpha=0.7)

 
plt.legend(loc='best')

 
if len(scores) > 10:
     
    midpoint = len(scores) // 2
    early_avg = np.mean(scores[:midpoint])
    late_avg = np.mean(scores[midpoint:])
    
     
         
         
         
         
         

 
plt.tight_layout()

 
plt.savefig('score_plot.png', dpi=300, bbox_inches='tight')

 
plt.show()

print("Plot saved as 'score_plot.png'")