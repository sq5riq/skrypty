# Test przepustowości pamięci RAM i VRAM (CuPy)

Skrypt w języku Python do pomiaru rzeczywistej szybkości transferu danych między pamięcią operacyjną (RAM) a pamięcią karty graficznej (VRAM) z wykorzystaniem biblioteki **CuPy** oraz wielowątkowości CPU.

Artykuł i wyniki testów na blogu: [Ai INFORMATYK](https://marcinnoga75.blogspot.com/2026/11/ai-informatyk.html)

## Co mierzy skrypt?
1. **RAM -> VRAM (PCIe H2D)** – wysyłanie danych z pamięci RAM do pamięci karty graficznej.
2. **VRAM -> RAM (PCIe D2H)** – pobieranie danych z karty graficznej do pamięci RAM.
3. **VRAM -> VRAM (GDDR D2D)** – kopiowanie danych bezpośrednio wewnątrz pamięci GPU.
4. **RAM -> RAM (CPU wielowątkowo)** – kopiowanie bloków pamięci w RAM przy użyciu wielu wątków procesora.

## Wymagania i instalacja
Do uruchomienia skryptu potrzebna jest karta graficzna **NVIDIA** (z obsługą CUDA) oraz zainstalowany **Python 3**.

Zainstaluj odpowiednią wersję biblioteki CuPy (dla CUDA 12.x lub CUDA 11.x):

```bash
pip install cupy-cuda12x
```
*(dla starszych sterowników z CUDA 11.x użyj `pip install cupy-cuda11x`)*

## Uruchomienie
Pobierz plik `test_cupy.py` i uruchom go w wierszu poleceń (CMD / PowerShell / Terminal):

```bash
python test_cupy.py
```
