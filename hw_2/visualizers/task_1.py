import pandas as pd
import matplotlib.pyplot as plt
import os

# Путь к файлу
csv_path = 'results/mandelbrot.csv'

def plot_mandelbrot():
    # Проверка наличия файла
    if not os.path.exists(csv_path):
        print(f"Ошибка: Файл {csv_path} не найден.")
        print("Убедитесь, что вы запустили C-программу и создали папку results.")
        return

    print(f"Чтение данных из {csv_path}...")
    
    try:
        # Читаем CSV
        df = pd.read_csv(csv_path)
        
        # Проверка, есть ли данные
        if df.empty:
            print("Файл пуст. Возможно, ни одна точка не попала в множество (проверьте логику C-кода).")
            return

        print(f"Загружено {len(df)} точек. Построение графика...")

        # Настройка размера фигуры (в дюймах)
        plt.figure(figsize=(10, 10), dpi=100)
        
        # Рисуем точки
        # s=0.1 — размер точки (чем больше точек, тем меньше должен быть этот параметр)
        # c='black' — цвет точек
        plt.scatter(df['x'], df['y'], s=0.1, c='black', marker='.')

        # Настройки осей
        plt.axis('equal')  # Чтобы пропорции не искажались
        plt.title('Множество Мандельброта')
        plt.xlabel('Re (x)')
        plt.ylabel('Im (y)')
        
        # Сохранение результата в файл картинки
        output_img = 'results/mandelbrot_plot.png'
        plt.savefig(output_img)
        print(f"График сохранен в файл: {output_img}")
        
        # Показать окно с графиком
        plt.show()

    except Exception as e:
        print(f"Произошла ошибка при обработке: {e}")

if __name__ == "__main__":
    plot_mandelbrot()