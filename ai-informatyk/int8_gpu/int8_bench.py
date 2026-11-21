#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
int8_bench.py — uniwersalny benchmark INT8 (wynik w TOPS)

Backendy (wykrywane automatycznie, każdy jest opcjonalny):

  1. OpenCL  (pakiet: pyopencl)
     ARM Mali, Qualcomm Adreno, NVIDIA, AMD, Intel, PoCL (CPU) ...
     Mierzy szczytową przepustowość iloczynu skalarnego INT8 (4 x int8 -> int32).
     Sam wybiera najszybszą instrukcję, jaką ma urządzenie:
        cl_khr_integer_dot_product          (standard Khronos)
        cl_arm_integer_dot_product_*        (Mali G7x/G5x/G6xx/G7xx ...)
        PTX dp4a                            (NVIDIA, sm_61+)
        zwykły kod OpenCL                   (fallback – działa wszędzie)
     Każda metoda jest najpierw sprawdzana pod kątem poprawności wyników.

  2. CUDA    (pakiet: torch z obsługą CUDA)
     NVIDIA, także Jetson (który nie ma OpenCL).
     Mnożenie macierzy INT8 na rdzeniach Tensor (torch._int_mm -> cuBLASLt),
     czyli realne TOPS porównywalne z danymi producenta.

Liczenie: 1 MAC (mnożenie + dodawanie) = 2 operacje.  1 TOPS = 10^12 op/s.

Użycie:
  python int8_bench.py                 # wszystko, co się da wykryć
  python int8_bench.py --list          # tylko lista urządzeń
  python int8_bench.py --backend opencl --device mali
  python int8_bench.py --json wynik.json
"""

import argparse
import contextlib
import json
import os
import platform
import statistics
import sys
import time
import warnings

os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
warnings.filterwarnings("ignore")

try:
    import numpy as np
except ImportError:
    print("Brak numpy. Zainstaluj:  pip install numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parametry kernela „peak”
# ---------------------------------------------------------------------------
NCH = 8            # niezależne łańcuchy akumulatorów na wątek (ukrywanie opóźnień)
UNROLL = 16        # rozwinięcie pętli
DOTS_PER_ITER = NCH * UNROLL
OPS_PER_DOT = 8    # 4 mnożenia + 4 dodawania
MAX_ITERS = 1 << 22

# ---------------------------------------------------------------------------
# Implementacje dot8(a, b, c) = c + sum(a[i]*b[i]),  a,b = 4 x int8 w uint
# ---------------------------------------------------------------------------
DOT_KHR = r"""
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
int dot8(uint a, uint b, int c) { return dot_acc_sat(as_char4(a), as_char4(b), c); }
"""

DOT_ARM_SAT = r"""
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_saturate_int8 : enable
int dot8(uint a, uint b, int c) { return arm_dot_acc_sat(as_char4(a), as_char4(b), c); }
"""

DOT_ARM_ACC = r"""
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
int dot8(uint a, uint b, int c) { return arm_dot_acc(as_char4(a), as_char4(b), c); }
"""

DOT_ARM = r"""
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
int dot8(uint a, uint b, int c) { return (int)((uint)c + (uint)arm_dot(as_char4(a), as_char4(b))); }
"""

DOT_NV_PTX = r"""
int dot8(uint a, uint b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
"""

DOT_GENERIC = r"""
int dot8(uint a, uint b, int c) {
    char4 x = as_char4(a), y = as_char4(b);
    int s = (int)x.s0 * (int)y.s0 + (int)x.s1 * (int)y.s1
          + (int)x.s2 * (int)y.s2 + (int)x.s3 * (int)y.s3;
    return (int)((uint)c + (uint)s);
}
"""


def kernel_body():
    init = []
    prev = "h"
    for j in range(NCH):
        init.append(f"    uint a{j} = {prev} * 1664525u + 1013904223u;  int c{j} = (int)(a{j} >> 9);")
        prev = f"a{j}"
    # Operand b = bieżący akumulator -> wartości zmieniają się w każdym kroku,
    # więc kompilator nie może „sprytnie” skrócić pętli.
    step = "\n".join(f"        c{j} = dot8(a{j}, (uint)c{j}, c{j});" for j in range(NCH))
    body = "\n".join([step] * UNROLL)
    xor = " ^ ".join(f"c{j}" for j in range(NCH))
    return f"""
__kernel void peak(__global int *out, const uint seed, const int iters)
{{
    const uint gid = (uint)get_global_id(0);
    uint h = (gid * 2654435761u) ^ seed;
{chr(10).join(init)}
    for (int i = 0; i < iters; ++i) {{
{body}
    }}
    out[gid] = {xor};
}}

__kernel void check(__global const uint *A, __global const uint *B,
                    __global const int *C, __global int *O)
{{
    const int i = get_global_id(0);
    O[i] = dot8(A[i], B[i], C[i]);
}}
"""


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------
def fmt_ops(ops_per_s):
    if ops_per_s >= 1e12:
        return f"{ops_per_s / 1e12:.2f} TOPS"
    if ops_per_s >= 1e9:
        return f"{ops_per_s / 1e9:.1f} GOPS ({ops_per_s / 1e12:.4f} TOPS)"
    return f"{ops_per_s / 1e6:.1f} MOPS ({ops_per_s / 1e12:.6f} TOPS)"


@contextlib.contextmanager
def quiet_stderr():
    """Ucisza komunikaty kompilatora OpenCL przy próbach kolejnych metod."""
    try:
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stderr.flush()
        os.dup2(devnull, fd)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


def hr(title=""):
    line = "─" * 72
    print(f"\n{line}\n {title}\n{line}" if title else line)


# ---------------------------------------------------------------------------
# Backend OpenCL
# ---------------------------------------------------------------------------
def opencl_list(name_filter=None):
    try:
        import pyopencl as cl
    except ImportError:
        return None, "brak pakietu pyopencl  (pip install pyopencl)"
    found = []
    try:
        platforms = cl.get_platforms()
    except Exception as e:
        return None, f"brak sterownika OpenCL ({e})"
    for p in platforms:
        try:
            devs = p.get_devices()
        except Exception:
            continue
        for d in devs:
            if name_filter and name_filter.lower() not in (d.name + " " + d.vendor).lower():
                continue
            found.append((p, d))
    return found, None


def opencl_methods(dev):
    ext = set(dev.extensions.split())
    vendor = (dev.vendor + " " + dev.platform.name).lower()
    m = []
    if "cl_khr_integer_dot_product" in ext:
        m.append(("khr dot_acc_sat", DOT_KHR, ["-cl-std=CL3.0", ""]))
    if "cl_arm_integer_dot_product_accumulate_saturate_int8" in ext:
        m.append(("arm_dot_acc_sat", DOT_ARM_SAT, [""]))
    if "cl_arm_integer_dot_product_accumulate_int8" in ext:
        m.append(("arm_dot_acc", DOT_ARM_ACC, [""]))
    if "cl_arm_integer_dot_product_int8" in ext:
        m.append(("arm_dot", DOT_ARM, [""]))
    if "nvidia" in vendor:
        m.append(("PTX dp4a", DOT_NV_PTX, [""]))
    m.append(("generic (fallback)", DOT_GENERIC, [""]))
    return m


def opencl_validate(cl, ctx, q, prg):
    n = 4096
    rng = np.random.default_rng(1)
    A = rng.integers(0, 2**32, n, dtype=np.uint32)
    B = rng.integers(0, 2**32, n, dtype=np.uint32)
    C = rng.integers(-10**6, 10**6, n, dtype=np.int32)
    ref = (A.view(np.int8).reshape(-1, 4).astype(np.int32)
           * B.view(np.int8).reshape(-1, 4).astype(np.int32)).sum(1).astype(np.int32) + C
    mf = cl.mem_flags
    bA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bC = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
    O = np.empty(n, dtype=np.int32)
    bO = cl.Buffer(ctx, mf.WRITE_ONLY, O.nbytes)
    k = cl.Kernel(prg, "check")
    k.set_args(bA, bB, bC, bO)
    cl.enqueue_nd_range_kernel(q, k, (n,), None)
    cl.enqueue_copy(q, O, bO)
    q.finish()
    return bool(np.array_equal(O, ref))


def opencl_bench_device(p, dev, args):
    import pyopencl as cl

    print(f"\n▶ OpenCL: {dev.name.strip()}")
    print(f"  producent : {dev.vendor.strip()}  |  platforma: {p.name.strip()}")
    print(f"  wersja    : {dev.version.strip()}  |  sterownik: {dev.driver_version.strip()}")
    print(f"  jednostki : {dev.max_compute_units} CU  |  zegar: {dev.max_clock_frequency} MHz")

    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    body = kernel_body()

    prg, method = None, None
    for name, dot_src, opts_list in opencl_methods(dev):
        for opts in opts_list:
            try:
                with quiet_stderr():
                    cand = cl.Program(ctx, dot_src + body).build(options=opts, devices=[dev])
                if opencl_validate(cl, ctx, q, cand):
                    prg, method = cand, name
                else:
                    print(f"  ✗ {name}: złe wyniki – pomijam")
                break
            except Exception:
                continue
        if prg is not None:
            break
    if prg is None:
        print("  ✗ nie udało się zbudować żadnego kernela")
        return None
    print(f"  metoda    : {method}  ✓ wyniki poprawne")

    knl = cl.Kernel(prg, "peak")
    maxwg = knl.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
    pref = knl.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
    lsize = min(256, maxwg)
    if pref and pref <= lsize:
        lsize = (lsize // pref) * pref
    cu = max(1, dev.max_compute_units)
    gsize = cu * 4096
    gsize = ((gsize + lsize - 1) // lsize) * lsize

    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, gsize * 4)

    def run(iters):
        knl.set_args(out, np.uint32(12345), np.int32(iters))
        t0 = time.perf_counter()
        ev = cl.enqueue_nd_range_kernel(q, knl, (gsize,), (lsize,))
        ev.wait()
        t1 = time.perf_counter()
        try:
            t = (ev.profile.end - ev.profile.start) * 1e-9
            if t <= 0:
                t = t1 - t0
        except Exception:
            t = t1 - t0
        return t

    # Autostrojenie liczby iteracji, żeby jeden pomiar trwał ~args.time s
    iters, t = 8, run(8)
    while t < args.time * 0.5 and iters < MAX_ITERS:
        iters = min(MAX_ITERS, max(iters * 2, int(iters * args.time / max(t, 1e-6))))
        t = run(iters)
    iters = max(1, min(MAX_ITERS, int(iters * args.time / max(t, 1e-6))))

    run(iters)  # rozgrzewka
    ops = gsize * iters * DOTS_PER_ITER * OPS_PER_DOT
    rates = []
    for _ in range(args.repeats):
        rates.append(ops / run(iters))
    best, med = max(rates), statistics.median(rates)
    print(f"  konfig.   : global={gsize}, local={lsize}, iteracje={iters}")
    print(f"  WYNIK     : szczyt {fmt_ops(best)}   |   mediana {fmt_ops(med)}")
    return {
        "backend": "OpenCL",
        "device": dev.name.strip(),
        "vendor": dev.vendor.strip(),
        "method": method,
        "test": "peak INT8 dot4",
        "tops_peak": best / 1e12,
        "tops_median": med / 1e12,
    }


# ---------------------------------------------------------------------------
# Backend CUDA (PyTorch, rdzenie Tensor)
# ---------------------------------------------------------------------------
def cuda_list(name_filter=None):
    try:
        import torch
    except ImportError:
        return None, "brak PyTorch  (pip install torch)"
    if not torch.cuda.is_available():
        return None, "PyTorch bez CUDA lub brak GPU NVIDIA"
    found = []
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        if name_filter and name_filter.lower() not in pr.name.lower():
            continue
        found.append((i, pr))
    return found, None


def cuda_bench_device(idx, pr, args):
    import torch

    dev = torch.device(f"cuda:{idx}")
    print(f"\n▶ CUDA: {pr.name}")
    print(f"  compute capability: {pr.major}.{pr.minor}  |  SM: {pr.multi_processor_count}"
          f"  |  pamięć: {pr.total_memory / 2**30:.1f} GiB  |  torch {torch.__version__}")
    if not hasattr(torch, "_int_mm"):
        print("  ✗ ta wersja PyTorch nie ma torch._int_mm (potrzebna 2.1+)")
        return None

    # Test poprawności
    try:
        a = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=dev)
        w = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=dev)
        got = torch._int_mm(a, w.t()).cpu().numpy()
        ref = a.cpu().numpy().astype(np.int64) @ w.cpu().numpy().astype(np.int64).T
        if not np.array_equal(got.astype(np.int64), ref):
            print("  ✗ złe wyniki INT8 GEMM – pomijam")
            return None
    except Exception as e:
        print(f"  ✗ INT8 GEMM niedostępny na tym GPU ({str(e).splitlines()[0][:80]})")
        return None

    # Duża macierz (zmniejszana, jeśli brakuje pamięci)
    n = args.gemm_size
    while n >= 512:
        try:
            A = torch.randint(-128, 127, (n, n), dtype=torch.int8, device=dev)
            W = torch.randint(-128, 127, (n, n), dtype=torch.int8, device=dev)
            Wt = W.t()
            torch._int_mm(A, Wt)
            torch.cuda.synchronize(dev)
            break
        except RuntimeError:
            A = W = Wt = None
            torch.cuda.empty_cache()
            n //= 2
    else:
        print("  ✗ za mało pamięci GPU")
        return None

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def run(reps):
        s.record()
        for _ in range(reps):
            torch._int_mm(A, Wt)
        e.record()
        torch.cuda.synchronize(dev)
        return s.elapsed_time(e) / 1e3

    t1 = run(3) / 3
    reps = max(1, int(args.time / max(t1, 1e-6)))
    run(reps)  # rozgrzewka
    ops = 2.0 * n * n * n * reps
    rates = [ops / run(reps) for _ in range(args.repeats)]
    best, med = max(rates), statistics.median(rates)
    print(f"  konfig.   : GEMM {n}x{n}x{n}, powtórzeń={reps}, metoda: torch._int_mm (Tensor Cores)")
    print(f"  WYNIK     : szczyt {fmt_ops(best)}   |   mediana {fmt_ops(med)}")
    return {
        "backend": "CUDA",
        "device": pr.name,
        "vendor": "NVIDIA",
        "method": "torch._int_mm / cuBLASLt",
        "test": f"INT8 GEMM {n}^3",
        "tops_peak": best / 1e12,
        "tops_median": med / 1e12,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Uniwersalny benchmark INT8 (TOPS)")
    ap.add_argument("--backend", choices=["auto", "opencl", "cuda"], default="auto")
    ap.add_argument("--device", help="filtr po nazwie urządzenia, np. mali, rtx")
    ap.add_argument("--time", type=float, default=0.25,
                    help="czas jednego pomiaru w s (domyślnie 0.25)")
    ap.add_argument("--repeats", type=int, default=5, help="liczba pomiarów (domyślnie 5)")
    ap.add_argument("--gemm-size", type=int, default=8192, help="rozmiar macierzy dla CUDA")
    ap.add_argument("--list", action="store_true", help="tylko wypisz urządzenia")
    ap.add_argument("--json", help="zapisz wyniki do pliku JSON")
    args = ap.parse_args()

    hr("INT8 BENCHMARK")
    print(f" system : {platform.system()} {platform.release()}  |  CPU/arch: {platform.machine()}")
    print(f" python : {platform.python_version()}  |  numpy {np.__version__}")

    ocl, ocl_err = (None, "pominięto") if args.backend == "cuda" else opencl_list(args.device)
    cud, cud_err = (None, "pominięto") if args.backend == "opencl" else cuda_list(args.device)

    hr("WYKRYTE URZĄDZENIA")
    if ocl:
        for p, d in ocl:
            print(f" [OpenCL] {d.name.strip()}  ({d.vendor.strip()})")
    else:
        print(f" [OpenCL] —  {ocl_err or 'nie znaleziono urządzeń'}")
    if cud:
        for i, pr in cud:
            print(f" [CUDA]   {pr.name}  (cc {pr.major}.{pr.minor})")
    else:
        print(f" [CUDA]   —  {cud_err or 'nie znaleziono urządzeń'}")

    if args.list:
        return
    if not ocl and not cud:
        print("\nBrak urządzeń do testu. Zainstaluj sterownik GPU oraz:")
        print("  pip install pyopencl        (Mali, AMD, Intel, NVIDIA ...)")
        print("  pip install torch           (NVIDIA – rdzenie Tensor, Jetson)")
        sys.exit(2)

    hr("POMIARY")
    results = []
    for p, d in ocl or []:
        try:
            r = opencl_bench_device(p, d, args)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ✗ błąd: {e}")
    for i, pr in cud or []:
        try:
            r = cuda_bench_device(i, pr, args)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ✗ błąd: {e}")

    hr("PODSUMOWANIE  (1 MAC = 2 operacje)")
    if not results:
        print(" brak wyników")
    else:
        w = max(len(r["device"]) for r in results)
        w = min(max(w, 10), 36)
        print(f" {'Urządzenie':<{w}}  {'Backend':<7}  {'Metoda':<26}  {'TOPS szczyt':>11}  {'mediana':>9}")
        for r in results:
            print(f" {r['device'][:w]:<{w}}  {r['backend']:<7}  {r['method'][:26]:<26}"
                  f"  {r['tops_peak']:>11.4f}  {r['tops_median']:>9.4f}")
        print("\n OpenCL = szczytowa przepustowość jednostek ALU (iloczyn skalarny int8).")
        print(" CUDA   = mnożenie macierzy na rdzeniach Tensor (NVIDIA).")

    if args.json:
        meta = {"system": platform.platform(), "machine": platform.machine(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n zapisano: {args.json}")


if __name__ == "__main__":
    main()
