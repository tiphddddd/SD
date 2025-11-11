import matplotlib.pyplot as plt
import numpy as np

def plot_agent_comparison(results_data, random_chance_baseline=None, time_threshold_sec=None, save_path=None):
    """
    Creates a side-by-side comparison of agent accuracy and time per task.
    
    Args:
        results_data (dict): A dictionary where keys are agent names (str)
                             and values are tuples (accuracies_list, times_list).
                             Example:
                             {
                                 "Agent One": ([0.5, 0.6], [10, 12]),
                                 "Agent Two": ([0.7, 0.8], [20, 22])
                             }
        
        random_chance_baseline (float, optional): Y-value for a horizontal line 
                                                  on the accuracy plot.
        
        time_threshold_sec (float, optional): Y-value for a horizontal line 
                                              on the time plot.
        
        save_path (str, optional): If provided, saves the figure to this path 
                                   instead of showing it.
    """
    
    num_agents = len(results_data)
    if num_agents == 0:
        print("Warning: No data provided to plot.")
        return

    agent_names = list(results_data.keys())
    
    # Get the number of tasks from the first agent's data
    try:
        first_agent_data = list(results_data.values())[0]
        num_tasks = len(first_agent_data[0]) # Length of accuracies_list
        tasks = np.arange(1, num_tasks + 1)
    except (IndexError, TypeError):
        print("Error: Data format is incorrect. Expected dict[str, (list, list)].")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # === 1. Accuracy Plot (Line Plot) ===
    
    # Use different markers for each agent
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for i, (agent_name, (accuracies, _)) in enumerate(results_data.items()):
        if len(accuracies) != num_tasks:
            print(f"Warning: Skipping agent '{agent_name}' due to mismatched task count.")
            continue
        
        style_marker = markers[i % len(markers)]
        ax1.plot(tasks, accuracies, marker=style_marker, linestyle='-', 
                 label=agent_name, alpha=0.8, linewidth=2)

    if random_chance_baseline is not None:
        ax1.axhline(y=random_chance_baseline, color='gray', linestyle='--', 
                    alpha=0.7, label=f'Random Chance ({random_chance_baseline*100:.0f}%)')
        
    ax1.set_xlabel('Task Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy per Task')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05]) # Give a little space at the top
    ax1.set_xticks(tasks) # Ensure integer task numbers

    # === 2. Time Plot (Grouped Bar Plot) ===
    
    # Calculate bar positions for grouped bar chart
    bar_width = 0.8 / num_agents # Total width of 0.8, divided by agents
    offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, num_agents)
    
    for i, (agent_name, (_, times)) in enumerate(results_data.items()):
        if len(times) != num_tasks:
            continue # Already warned from accuracy plot
            
        ax2.bar(tasks + offsets[i], times, bar_width, 
                label=agent_name, alpha=0.7)

    if time_threshold_sec is not None:
        ax2.axhline(y=time_threshold_sec, color='red', linestyle='--', 
                    alpha=0.7, label=f'{time_threshold_sec}s threshold')

    ax2.set_xlabel('Task Number')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training + Prediction Time per Task')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3) # Grid on y-axis is cleaner for bars
    ax2.set_xticks(tasks)

    # === Final Touches ===
    plt.suptitle('Agent Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# -----------------------------------------------------------------
# This allows you to run `python utils/visualization.py` to test it
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("Testing plot_agent_comparison function...")
    
    # Create some fake data for testing
    tasks = 5
    random_acc = np.random.uniform(0.05, 0.15, tasks)
    random_t = np.random.uniform(5, 10, tasks)
    
    linear_acc = np.array([0.3, 0.4, 0.35, 0.42, 0.38]) + np.random.rand(tasks)*0.05
    linear_t = np.random.uniform(20, 30, tasks)
    
    mlp_acc = np.array([0.6, 0.65, 0.7, 0.68, 0.72]) + np.random.rand(tasks)*0.05
    mlp_t = np.random.uniform(60, 90, tasks)
    
    # This is the dictionary structure!
    test_results = {
        "Random Agent": (random_acc, random_t),
        "Linear Agent": (linear_acc, linear_t),
        "MLP Agent": (mlp_acc, mlp_t)
    }
    
    plot_agent_comparison(test_results, 
                          random_chance_baseline=0.1, 
                          time_threshold_sec=60,
                          save_path="test_plot.png")