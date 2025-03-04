import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

df = pd.read_csv('rps.csv')

steps = df['Step']   
rps_values = df['Value']    

plt.figure(figsize=(12, 6))

plt.scatter(steps, rps_values, color='lightcoral', s=10, alpha=0.4, 
           label='Reward Per Step')

window_size = min(51, len(rps_values) - 1 if len(rps_values) % 2 == 0 else len(rps_values))   
if window_size > 3:   
    poly_order = 3     
    try:
        smoothed_values = savgol_filter(rps_values, window_size, poly_order)
        plt.plot(steps, smoothed_values, '-', color='red', linewidth=2, 
                label=f'Smoothed Trend (window={window_size})')
    except Exception as e:
        print(f"Could not apply Savgol filter: {e}")

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

ma_window = min(30, len(rps_values) // 3)   
if ma_window >= 2:   
    ma_smoothed = moving_average(rps_values, ma_window)
    ma_steps = steps[ma_window-1:]
    plt.plot(ma_steps, ma_smoothed, '-', color='darkred', linewidth=2, 
            label=f'Moving Average (window={ma_window})')

if len(rps_values) > 0:
    mean_rps = np.mean(rps_values)
    max_rps = np.max(rps_values)
    min_rps = np.min(rps_values)
    
    plt.axhline(y=mean_rps, color='blue', linestyle='--', 
               label=f'Mean RPS: {mean_rps:.2f}')
    
    max_step = steps[np.argmax(rps_values)]
    plt.scatter(max_step, max_rps, color='gold', s=100, zorder=5, 
               label=f'Max RPS: {max_rps:.2f}')
    
    min_step = steps[np.argmin(rps_values)]
    plt.scatter(min_step, min_rps, color='purple', s=100, zorder=5, 
               label=f'Min RPS: {min_rps:.2f}')
    
    plt.annotate(f'Max: {max_rps:.2f}',
                xy=(max_step, max_rps),
                xytext=(max_step, max_rps + (max_rps - min_rps)*0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=10)
    
    first_quarter_avg = np.mean(rps_values[:len(rps_values)//4])
    last_quarter_avg = np.mean(rps_values[3*len(rps_values)//4:])
    trend_direction = "increasing" if last_quarter_avg > first_quarter_avg else "decreasing"
    
    plt.annotate(f"Overall trend: {trend_direction}",
                xy=(steps[len(steps)//2], mean_rps),
                xytext=(steps[len(steps)//2], mean_rps + (max_rps - min_rps)*0.25),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=12)

if min_rps < 0 and max_rps > 0:
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.xlabel('Training Episode', fontsize=14)
plt.ylabel('Reward Per Step (RPS)', fontsize=14)
plt.title('Reward Per Step During Training', fontsize=16)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='best')

plt.tight_layout()

plt.savefig('reward_per_step_analysis.png', dpi=300, bbox_inches='tight')

plt.show()

print("Plot saved as 'reward_per_step_analysis.png'")