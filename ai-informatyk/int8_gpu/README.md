Claude Opus 5.5 poziom Extra
"Wygeneruj kompletny kod uniwersalnego benchmarku INT8, 
działającego na różnych platformach i układach graficznych, m.in. ARM/Mali i NVIDIA. 
Program ma automatycznie wykrywać urządzenie, mierzyć szybkość obliczeń i podawać wynik w TOPS."
===============================================================================================================================================================
Piszę skrypt w Pythonie (OpenCL dla Mali/NVIDIA/AMD/Intel + CUDA Tensor Cores dla NVIDIA), potem testuję go tutaj.
Działa (ścieżka uniwersalna). Sprawdzam jeszcze, czy przełączanie metod działa, gdy instrukcja nie istnieje albo daje złe wyniki.
Przełączanie działa. Wyciszam jeszcze komunikaty kompilatora przy próbach.

Skrypt jest gotowy: int8_bench.py. Sam wykrywa układ, wybiera dla niego najszybszą instrukcję INT8, sprawdza, czy liczy poprawnie, i podaje wynik w TOPS.

*Sprawdziłam go tylko na procesorze* (przez OpenCL), bo tutaj nie ma karty graficznej. Na nim działa:
- wykrywanie urządzenia,
- sprawdzanie poprawności wyników,
- wynik w TOPS,
- przejście na wolniejszą metodę, gdy szybsza nie działa.

Szybkie ścieżki dla Mali i NVIDIA nie były uruchomione na prawdziwym sprzęcie. Jeśli któraś nie zadziała, skrypt sam przejdzie na wolniejszą metodę, która działa wszędzie.

*Co mierzy:*
- *OpenCL* (Mali, Adreno, NVIDIA, AMD, Intel): szczytową szybkość obliczeń INT8 w zwykłych rdzeniach GPU. Sam wybiera instrukcję:
  - standardową cl_khr_integer_dot_product,
  - arm_dot na Mali,
  - dp4a na NVIDIA,
  - zwykły kod, gdy nic z tego nie ma.
- *CUDA* (NVIDIA, także Jetson): mnożenie macierzy INT8 na rdzeniach Tensor. To ten wynik zwykle podaje producent i na NVIDIA jest kilka razy wyższy niż w OpenCL.

*Uruchomienie: płytka z Mali (np. Orange Pi, Linux)*
- Otwórz terminal.
- sudo apt install python3-pip ocl-icd-libopencl1 clinfo
- Zainstaluj sterownik OpenCL dla Mali (libmali od producenta płytki albo Mesa Rusticl).
- pip install numpy pyopencl
- python3 int8_bench.py

*Uruchomienie: komputer z NVIDIA*
- Zainstaluj aktualny sterownik NVIDIA (ma już OpenCL).
- pip install numpy pyopencl
- Zainstaluj PyTorch z CUDA. Na Linuksie wystarczy pip install torch. Na Windowsie wybierz polecenie dla wersji CUDA na pytorch.org.
- python int8_bench.py

*Jetson:* nie ma OpenCL. Zainstaluj PyTorch od NVIDIA (JetPack), wtedy zadziała część CUDA.

*Przydatne opcje:*
- --list: tylko pokazuje znalezione urządzenia.
- --device mali: testuje tylko urządzenia z „mali” w nazwie.
- --json wynik.json: zapisuje wyniki do pliku.
- --time 0.5 --repeats 10: dłuższy i dokładniejszy pomiar.

*Czego nie mierzy:* układów NPU (np. NPU 6 TOPS w RK3588) ani Metal na Macach. Na Macu z Apple Silicon zadziała tylko wolniejsza metoda przez OpenCL.
