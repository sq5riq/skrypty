#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test wydajnosci INT8 na karcie NVIDIA.

Skrypt symuluje najciezsza czesc pracy modelu AI: mnozenie macierzy INT8
(GEMM) wykonywane wielokrotnie na GPU. Wynik podawany jest w TOPS.

Wymagania:
    pip install torch

Uruchomienie:
    python test_int8_nvidia.py
    python test_int8_nvidia.py --size 8192 --seconds 15
    python test_int8_nvidia.py --size 4096 --seconds 30 --monitor

Uwaga: to mikrobenchmark GEMM, a nie pomiar predkosci calego modelu.
"""

import argparse
import shutil
import subprocess
import threading
import time

import torch


def gpu_info():
    print("=" * 72)
    print(" TEST WYDAJNOSCI INT8 NVIDIA - SYMULACJA OBCIAZENIA MODELU AI")
    print("=" * 72)
    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA w PyTorch:   {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemExit("BLAD: PyTorch nie wykrywa karty NVIDIA/CUDA.")

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"GPU:              {props.name}")
    print(f"Pamiec VRAM:      {props.total_memory / 1024**3:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print("=" * 72)


def monitor_nvidia_smi(stop_event, interval=1.0):
    """Pokazuje uzycie GPU, VRAM, moc i temperature podczas testu."""
    if not shutil.which("nvidia-smi"):
        print("Monitor: nie znaleziono programu nvidia-smi.")
        return

    query = (
        "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    )
    while not stop_event.is_set():
        try:
            result = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits", "-i", "0"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().splitlines()[0]
            util, used, total, power, temp = [x.strip() for x in result.split(",")]
            print(
                f"[GPU] obciazenie: {util:>3}% | VRAM: {used}/{total} MiB | "
                f"moc: {power} W | temp.: {temp} C"
            )
        except Exception:
            pass
        stop_event.wait(interval)


def int8_matmul(a, b):
    """INT8 x INT8 z akumulacja INT32, tak jak w typowych kernelach AI."""
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError(
            "Ta wersja PyTorch nie udostepnia torch._int_mm. "
            "Zainstaluj aktualny PyTorch z obsluga CUDA."
        )
    return torch._int_mm(a, b)


def benchmark(size, seconds, warmup, monitor):
    device = torch.device("cuda")

    try:
        a = torch.randint(-128, 127, (size, size), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (size, size), device=device, dtype=torch.int8)
        c = torch.empty((size, size), device=device, dtype=torch.int32)
        del c
    except torch.cuda.OutOfMemoryError:
        free_mem, total_mem = torch.cuda.mem_get_info()
        raise SystemExit(
            f"Za malo VRAM dla macierzy {size}x{size}. "
            f"Wolne: {free_mem / 1024**3:.2f} GB. Uzyj np. --size 4096."
        )

    print(f"Rozmiar macierzy: {size} x {size}")
    print(f"Czas pomiaru:     {seconds:.1f} s")
    print(f"Rozgrzewka:       {warmup} iteracji")
    print("Rozgrzewanie GPU...")

    for _ in range(warmup):
        out = int8_matmul(a, b)
    torch.cuda.synchronize()

    stop_event = threading.Event()
    thread = None

    if monitor:
        thread = threading.Thread(
            target=monitor_nvidia_smi,
            args=(stop_event,),
            daemon=True
        )
        thread.start()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iterations = 0
    wall_start = time.perf_counter()

    start_event.record()

    while True:
        out = int8_matmul(a, b)
        torch.cuda.synchronize()
        iterations += 1

        if time.perf_counter() - wall_start >= seconds:
            break

    end_event.record()
    torch.cuda.synchronize()

    elapsed_s = start_event.elapsed_time(end_event) / 1000.0

    stop_event.set()

    if thread:
        thread.join(timeout=2.0)

    # Dla GEMM MxK razy KxN: M*N*K mnozen i tyle samo dodawan.
    operations = iterations * 2 * (size ** 3)
    tops = operations / elapsed_s / 1e12
    ms_per_iter = elapsed_s * 1000.0 / iterations

    # Odczyt jednej wartosci zapobiega calkowitemu pominieciu wyniku.
    checksum = int(out[0, 0].item())

    print("\n" + "-" * 72)
    print(" WYNIK")
    print("-" * 72)
    print(f"Liczba iteracji:  {iterations}")
    print(f"Czas GPU:         {elapsed_s:.3f} s")
    print(f"Czas iteracji:    {ms_per_iter:.3f} ms")
    print(f"Wydajnosc INT8:   {tops:.2f} TOPS")
    print(f"Wynik kontrolny:  {checksum}")
    print("-" * 72)
    print("TOPS = biliony operacji (mnozenie + dodawanie) na sekunde.")
    print("Wynik dotyczy samego GEMM INT8, nie szybkosci generowania tokenow.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark/symulacja obliczen INT8 na GPU NVIDIA"
    )
    parser.add_argument("--size", type=int, default=4096,
                        help="rozmiar macierzy, domyslnie 4096")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="czas testu w sekundach, domyslnie 10")
    parser.add_argument("--warmup", type=int, default=5,
                        help="liczba iteracji rozgrzewki, domyslnie 5")
    parser.add_argument("--monitor", action="store_true",
                        help="pokazuj parametry z nvidia-smi podczas testu")
    args = parser.parse_args()

    if args.size < 256:
        parser.error("--size musi wynosic co najmniej 256")
    if args.seconds <= 0:
        parser.error("--seconds musi byc wieksze od zera")
    if args.warmup < 0:
        parser.error("--warmup nie moze byc ujemne")

    gpu_info()
    benchmark(args.size, args.seconds, args.warmup, args.monitor)


if __name__ == "__main__":
    main()
