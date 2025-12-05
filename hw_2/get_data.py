import random
import math
import sys

def generate_nbody_file(filename, n_particles):
    """
    Генерирует файл для задачи N тел.
    Формат:
    N
    m x y z vx vy vz
    ...
    """
    
    # Константы для стабильной орбиты
    G = 6.67430e-11
    CENTER_MASS = 1.0e15  # Тяжелый центр
    MAX_RADIUS = 1000.0
    
    with open(filename, 'w') as f:
        # Первая строка: количество частиц
        f.write(f"{n_particles}\n")
        
        # 1. Создаем "Черную дыру" / Звезду в центре (чтобы система не разлетелась)
        # m, x, y, z, vx, vy, vz
        f.write(f"{CENTER_MASS} 0.0 0.0 0.0 0.0 0.0 0.0\n")
        
        # 2. Генерируем остальные N-1 частиц на орбите
        for _ in range(n_particles - 1):
            mass = random.uniform(1.0, 100.0)
            
            # Случайный радиус и угол
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(100.0, MAX_RADIUS)
            
            # Координаты (плоский диск для красоты, z немного варьируется)
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            z = random.uniform(-10.0, 10.0)
            
            # Расчет скорости для круговой орбиты: v = sqrt(G * M / r)
            # Это нужно, чтобы частицы красиво крутились, а не улетали в бесконечность
            velocity = math.sqrt(G * CENTER_MASS / dist)
            
            # Вектор скорости перпендикулярен радиусу
            vx = -velocity * math.sin(angle)
            vy = velocity * math.cos(angle)
            vz = random.uniform(-0.1, 0.1)
            
            f.write(f"{mass} {x} {y} {z} {vx} {vy} {vz}\n")

    print(f"Файл '{filename}' успешно создан. Количество частиц: {n_particles}")

if __name__ == "__main__":
    # Настройки по умолчанию
    N = 2000  # 2000 - хорошее число для начала. Если слишком быстро - ставьте 5000.
    FILE = "data/part_2/atom_big.txt"
    
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        FILE = sys.argv[2]
        
    generate_nbody_file(FILE, N)