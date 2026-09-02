
from __future__ import annotations

import platform
import subprocess
import ctypes
import os

def get_cuda_stats():
    """Checks for CUDA via nvidia-smi with cross-platform string formatting."""
    try:
        # Standard cross-platform invocation of nvidia-smi
        cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        is_windows = platform.system() == "Windows"
        
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, shell=is_windows).decode('utf-8', errors='ignore').strip()
        if not output:
            return None
            
        name, util, mem_used, mem_total = [x.strip() for x in output.split(',')]
        return {
            "type": f"CUDA ({platform.system()})",
            "name": name,
            "load": f"{util}%",
            "memory": f"{mem_used} MB / {mem_total} MB"
        }
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None

def get_cpu_stats():
    """Cross-platform memory tool avoiding heavy external modules."""
    sys_type = platform.system()
    total_mem, free_mem = 0, 0
    load_str = "N/A"

    try:

        #memory stats 
        if sys_type == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_mem, free_mem = stat.ullTotalPhys, stat.ullAvailPhys
            load_str = f"{stat.dwMemoryLoad}%"

        elif sys_type == "Darwin":  # macOS
            total_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            load_raw = os.getloadavg()[0]
            cores = os.cpu_count() or 1
            load_str = f"{min(100, int((load_raw / cores) * 100))}%"
            # Memory estimation fallback for speed
            vm = subprocess.check_output(["vm_stat"]).decode('utf-8')
            page_size = 4096
            free_pages = 0
            for line in vm.split('\n'):
                if "Pages free:" in line:
                    free_pages = int(line.split()[-1].replace('.', ''))
            free_mem = free_pages * page_size

        else:  # Linux / Unix fallback
            with open('/proc/meminfo', 'r') as f:
                mem_data = {l.split()[0].rstrip(':'): int(l.split()[1]) * 1024 for l in f if l.split()}
            total_mem = mem_data.get('MemTotal', 0)
            free_mem = mem_data.get('MemAvailable', mem_data.get('MemFree', 0))
            load_raw = os.getloadavg()[0]
            cores = os.cpu_count() or 1
            load_str = f"{min(100, int((load_raw / cores) * 100))}%"

        used_gb = (total_mem - free_mem) / (1024**3)
        total_gb = total_mem / (1024**3)
        mem_str = f"{used_gb:.2f} GB / {total_gb:.2f} GB"

    except Exception:
        mem_str, load_str = "Unknown", "Unknown"

    return {
        "type": f"CPU ({sys_type})",
        "load": load_str,
        "memory": mem_str
    }

