# Быстрый старт

## Установка зависимостей

### macOS
```bash
brew install openmpi
# или
brew install mpich
```

### Ubuntu/Debian
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
# или
sudo apt-get install libmpich-dev mpich
```

## Компиляция

```bash
make
```

## Быстрое тестирование

### Задание 1: Monte Carlo π
```bash
mpirun -np 4 ./task1_monte_carlo_pi 1000000
```

### Задание 2: Умножение матрицы на вектор
```bash
# По строкам
mpirun -np 4 ./task2_matrix_vector 100 r

# По столбцам
mpirun -np 4 ./task2_matrix_vector 100 c

# По блокам (требует квадратное число процессов)
mpirun -np 4 ./task2_matrix_vector 100 b
```

### Задание 3: Алгоритм Кэннона
```bash
# Требует квадратное число процессов и размер, кратный sqrt(процессов)
mpirun -np 4 ./task3_cannon_matrix_mult 100
```

### Задание 4: Задача Дирихле
```bash
# Требует квадратное число процессов и (размер-2), кратное sqrt(процессов)
mpirun -np 4 ./task4_dirichlet 50 0.0
```

## Запуск экспериментов

```bash
./run_experiments.sh
```

Этот скрипт автоматически запустит все задания с различными параметрами.

## Построение графиков

```bash
python3 plot_results.py
```

Графики будут сохранены как `task1_plots.png`, `task2_plots.png`, и т.д.

## Требования Python

```bash
pip3 install numpy matplotlib
```

