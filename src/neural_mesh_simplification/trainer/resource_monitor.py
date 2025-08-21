import time

import psutil
import torch


def monitor_resources(stop_event, main_pid):
    main_process = psutil.Process(main_pid)

    interval_idx = 0

    while not stop_event.is_set():
        try:
            # Get CPU and memory usage for the main process
            cpu_percent = main_process.cpu_percent(interval=1)
            memory_info = main_process.memory_info()
            total_memory_rss = memory_info.rss  # Start with main process memory usage

            # Include children processes
            for child in main_process.children(recursive=True):
                try:
                    cpu_percent += child.cpu_percent(interval=None)
                    child_memory = child.memory_info()
                    total_memory_rss += (
                        child_memory.rss
                    )  # Add child process memory usage
                except psutil.NoSuchProcess:
                    pass  # Child process no longer exists

            memory_usage_mb = total_memory_rss / (1024 * 1024)  # Convert to MB

            interval_idx += 1

            output = f"\rCPU: {cpu_percent:.1f}% | Memory: {memory_usage_mb:.2f} MB"

            # Get GPU and GPU memory usage
            if torch.cuda.is_available():
                gpu_info = []
                mem_total = 14.74 * 1024**3
                for i in range(torch.cuda.device_count()):
                    gpu_util = torch.cuda.utilization(i)  # GPU Utilization
                    mem_alloc = torch.cuda.memory_allocated(i) / (
                        1024 * 1024
                    )  # Convert to MB
                    # mem_total = torch.cuda.get_device_properties(i).total_memory / (
                        # 1024 * 1024
                    # )  # Total VRAM
                    gpu_info.append(
                        f"GPU {i}: {gpu_util:.1f}% | Mem: {mem_alloc:.2f}/{mem_total:.2f} MB"
                    )

                    output += " | " + " | ".join(gpu_info)

            if interval_idx % 3 == 0:
                output += "\n"

            print(output, end="", flush=True)

        except psutil.NoSuchProcess:
            print("\nMain process has terminated.")
            break
        time.sleep(1)
