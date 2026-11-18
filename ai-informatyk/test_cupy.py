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

free_mem_before, total_mem = cp.cuda.runtime.memGetInfo()

# Konfiguracja testu
size_gb = 1.0
iterations = 100
total_gb = size_gb * iterations

num_elements = int((size_gb * 1024**3) // 4)
size_bytes = num_elements * 4

print(
    f"Konfiguracja: blok {size_gb:.1f} GB x "
    f"{iterations} powtórzeń = {total_gb:.1f} GB na etap"
)
print(f"Testowany CPU: {cpu_name} ({cpu_threads} wątków)")
print(f"Testowane GPU: {device_name}")
print(f"GPU RAM: {total_mem/1024**3:.2f} GB")
print(f"Wolne:   {free_mem_before/1024**3:.2f} GB")
print("Alokacja pamięci RAM oraz VRAM... Proszę czekać.")

# Alokacja pamięci RAM (pinned memory dla GPU)
pinned_mem = cp.cuda.alloc_pinned_memory(size_bytes)
cpu_array = np.frombuffer(
    pinned_mem,
    dtype=np.float32,
    count=num_elements,
)
cpu_array.fill(1.0)

# Alokacja GPU
gpu_array = cp.empty(
    num_elements,
    dtype=cp.float32,
)

cp.cuda.Stream.null.synchronize()

free_mem_after, _ = cp.cuda.runtime.memGetInfo()

print(f"Wolne:   {free_mem_after/1024**3:.2f} GB")

used_vram = (
    free_mem_before - free_mem_after
) / 1024**3

print(f"Zajęto VRAM: {used_vram:.2f} GB")
print()

start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

# ==============================================================================
# TEST 1: RAM -> VRAM (Host to Device)
# ==============================================================================
print("-" * 58)
print(f"1. Rozpoczynam test: RAM -> VRAM ({total_gb:.0f} GB)...")
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
print(f"2. Rozpoczynam test: VRAM -> RAM ({total_gb:.0f} GB)...")
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
# TEST 3: VRAM -> VRAM (Wewnętrzna pamięć karty - GDDR)
# ==============================================================================
print("-" * 58)

# Faktycznie przetwarzana ilość danych
total_gb_d2d = (gpu_array.nbytes * iterations) / (1024**3)

print(f"3. Rozpoczynam test: VRAM -> VRAM ({total_gb_d2d:.1f} GB)...")

gpu_dst = cp.empty_like(gpu_array)

# Rozgrzewka
cp.copyto(gpu_dst, gpu_array)
cp.cuda.Stream.null.synchronize()

start_event.record()

for _ in range(iterations):
    cp.copyto(gpu_dst, gpu_array)

end_event.record()
end_event.synchronize()

time_d2d = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

bw_d2d = total_gb_d2d / time_d2d

print(f"   Czas: {time_d2d:.3f} s | Prędkość: {bw_d2d:.2f} GB/s")

del gpu_dst
cp.get_default_memory_pool().free_all_blocks()

# ==============================================================================
# TEST 4: RAM -> RAM (Wielowątkowe kopiowanie w pamięci operacyjnej)
# ==============================================================================
print("-" * 58)
print(f"4. Rozpoczynam test: RAM -> RAM ({total_gb:.0f} GB, {cpu_threads} wątków CPU)...")

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
print(f"  VRAM -> VRAM (GDDR D2D):    {bw_d2d:6.2f} GB/s")
print("=" * 58)
