from concurrent.futures import ThreadPoolExecutor
import os
import platform
import time
import cupy as cp
import numpy as np


# Funkcja pobierająca pełną nazwę procesora w Windows (lub Linux)
def get_cpu_name():
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name.strip()
        except Exception:
            pass
    return platform.processor() or "Nieznany procesor"


# Sprawdzenie karty i CPU
device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
cpu_name = get_cpu_name()
cpu_threads = os.cpu_count() or 4

print(f"Testowane GPU: {device_name}")
print(f"Testowany CPU: {cpu_name} ({cpu_threads} wątków)")

# Konfiguracja testu: blok 4 GB x 50 powtórzeń = 200 GB na każdy test
size_gb = 4.0
iterations = 50
total_gb = size_gb * iterations

num_elements = int((size_gb * 1024**3) // 4)
size_bytes = num_elements * 4

print(
    f"Konfiguracja: blok {size_gb} GB x {iterations} powtórzeń = {total_gb} GB"
    " na etap"
)
print("Alokacja pamięci RAM oraz VRAM... Proszę czekać.\n")

# Alokacja pamięci RAM (pinned memory dla GPU)
pinned_mem = cp.cuda.alloc_pinned_memory(size_bytes)
cpu_array = np.frombuffer(pinned_mem, dtype=np.float32, count=num_elements)
cpu_array.fill(1.0)

free_mem, total_mem = cp.cuda.Device(0).mem_info
gpu_array = cp.empty(num_elements, dtype=cp.float32)

start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

# ==============================================================================
# TEST 1: RAM -> VRAM (Host to Device)
# ==============================================================================
print("-" * 58)
print("1. Rozpoczynam test: RAM -> VRAM (200 GB)...")
gpu_array.set(cpu_array)
cp.cuda.Stream.null.synchronize()

start_event.record()
for _ in range(iterations):
    gpu_array.set(cpu_array)
end_event.record()
end_event.synchronize()

time_h2d = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
bw_h2d = total_gb / time_h2d
print(f"   Czas: {time_h2d:.3f} s | Prędkość: {bw_h2d:.2f} GB/s")

# ==============================================================================
# TEST 2: VRAM -> RAM (Device to Host)
# ==============================================================================
print("-" * 58)
print("2. Rozpoczynam test: VRAM -> RAM (200 GB)...")
gpu_array.get(out=cpu_array)
cp.cuda.Stream.null.synchronize()

start_event.record()
for _ in range(iterations):
    gpu_array.get(out=cpu_array)
end_event.record()
end_event.synchronize()

time_d2h = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
bw_d2h = total_gb / time_d2h
print(f"   Czas: {time_d2h:.3f} s | Prędkość: {bw_d2h:.2f} GB/s")

# ==============================================================================
# TEST 3: VRAM -> VRAM (Wewnętrzna pamięć karty - GDDR6)
# ==============================================================================
print("-" * 58)
print("3. Rozpoczynam test: VRAM -> VRAM (200 GB)...")

if free_mem < 8.5 * (1024**3):
    del gpu_array
    cp.get_default_memory_pool().free_all_blocks()
    vram_elements = int((2.0 * 1024**3) // 4)
    vram_iters = 100
    gpu_src = cp.ones(vram_elements, dtype=cp.float32)
    gpu_dst = cp.empty(vram_elements, dtype=cp.float32)
else:
    vram_iters = iterations
    gpu_src = gpu_array
    gpu_dst = cp.empty(num_elements, dtype=cp.float32)

cp.copyto(gpu_dst, gpu_src)
cp.cuda.Stream.null.synchronize()

start_event.record()
for _ in range(vram_iters):
    cp.copyto(gpu_dst, gpu_src)
end_event.record()
end_event.synchronize()

time_d2d = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
bw_d2d = total_gb / time_d2d
print(f"   Czas: {time_d2d:.3f} s | Prędkość: {bw_d2d:.2f} GB/s")

# Zwalniamy VRAM przed testem czystego RAM-u
del gpu_src, gpu_dst
cp.get_default_memory_pool().free_all_blocks()

# ==============================================================================
# TEST 4: RAM -> RAM (Wielowątkowe kopiowanie w pamięci operacyjnej)
# ==============================================================================
print("-" * 58)
print(f"4. Rozpoczynam test: RAM -> RAM (200 GB, {cpu_threads} wątków CPU)...")

cpu_dst = np.empty(num_elements, dtype=np.float32)
cpu_dst.fill(0.0)

src_chunks = np.array_split(cpu_array, cpu_threads)
dst_chunks = np.array_split(cpu_dst, cpu_threads)


def copy_chunk(pair):
    np.copyto(pair[1], pair[0])  # pair[1] to cel (dst), pair[0] to źródło (src)


with ThreadPoolExecutor(max_workers=cpu_threads) as executor:
    # Rozgrzewka
    list(executor.map(copy_chunk, zip(src_chunks, dst_chunks)))

    t0 = time.perf_counter()
    for _ in range(iterations):
        list(executor.map(copy_chunk, zip(src_chunks, dst_chunks)))
    t1 = time.perf_counter()

time_ram = t1 - t0
bw_ram = total_gb / time_ram
print(f"   Czas: {time_ram:.3f} s | Prędkość: {bw_ram:.2f} GB/s")

# ==============================================================================
# PODSUMOWANIE
# ==============================================================================
print("\n" + "=" * 58)
print(f"PODSUMOWANIE WYDAJNOŚCI")
print(f"CPU: {cpu_name}")
print(f"GPU: {device_name}")
print("-" * 58)
print(f"  RAM  -> RAM  (CPU {cpu_threads:2d} wątków): {bw_ram:6.2f} GB/s")
print(f"  RAM  -> VRAM (PCIe H2D):     {bw_h2d:6.2f} GB/s")
print(f"  VRAM -> RAM  (PCIe D2H):     {bw_d2h:6.2f} GB/s")
print(f"  VRAM -> VRAM (GDDR6 D2D):    {bw_d2d:6.2f} GB/s")
print("=" * 58)
