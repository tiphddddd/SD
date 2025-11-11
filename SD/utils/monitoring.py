# utils/monitoring.py

import os
import psutil

def get_system_status():
    """
    Get the system resource status (CPU, Memory) and current process memory.
    """
    # Number of CPU cores
    num_cores = os.cpu_count()

    # System memory
    mem = psutil.virtual_memory()
    total_mem_gb = mem.total / (1024 ** 3)   # Convert bytes to GB
    used_mem_gb = mem.used / (1024 ** 3)     # Convert bytes to GB
    free_mem_gb = mem.available / (1024 ** 3)  # Convert bytes to GB

    # Current Python process memory usage
    process = psutil.Process(os.getpid()) # Get PID of the current process
    proc_mem_gb = process.memory_info().rss / (1024 ** 3) # RSS in GB

    status = {
        "cpu_cores": num_cores,
        "total_mem_gb": total_mem_gb,
        "used_mem_gb": used_mem_gb,
        "free_mem_gb": free_mem_gb,
        "process_mem_gb": proc_mem_gb
    }
    return status

def print_system_status():
    """
    Print the formatted system resource status.
    """
    try:
        status = get_system_status()
        
        print("=== System Status ===")
        print(f"CPU cores: {status['cpu_cores']}")
        print(f"Total RAM: {status['total_mem_gb']:.2f} GB | "
              f"Used: {status['used_mem_gb']:.2f} GB | "
              f"Free: {status['free_mem_gb']:.2f} GB")
        print(f"Current Python process memory: {status['process_mem_gb']:.2f} GB")
        print("=======================")

    except Exception as e:
        print(f"Error getting system status: {e}")

# -----------------------------------------------------------------
# Best practice: Allow this file to be both imported and run directly for testing.
# -----------------------------------------------------------------
if __name__ == "__main__":
    # This code block will execute if you run: `python utils/monitoring.py`
    print("Testing monitoring functions...")
    print_system_status()