import pandas as pd
import plotly.graph_objects as go
import sys
import os

def plot_trajectories(csv_filename):
    # Проверка наличия файла
    if not os.path.exists(csv_filename):
        print(f"Ошибка: Файл '{csv_filename}' не найден.")
        return

    filename = ".".join(csv_filename.split('/')[-1].split('.')[:-1])

    print(f"Загрузка данных из {csv_filename}...")
    
    # Чтение CSV файла. 
    # header=None, так как в выводе C программы нет заголовка с названиями колонок
    try:
        df = pd.read_csv(csv_filename, header=None)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # Определение количества тел
    # Формат: t, x1, y1, z1, x2, y2, z2, ...
    # Количество колонок = 1 + 3 * N
    num_columns = df.shape[1]
    n_bodies = (num_columns - 1) // 3
    
    print(f"Найдено тел: {n_bodies}")
    print(f"Количество шагов времени: {df.shape[0]}")

    fig = go.Figure()

    # Цветовая палитра (Plotly выберет автоматически, но можно задать свою)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

    for i in range(n_bodies):
        # Индексы колонок для тела i (нумерация с 0)
        # Колонка 0 - это время t
        # Тело 1: 1, 2, 3
        # Тело 2: 4, 5, 6
        col_x = 1 + i * 3
        col_y = 2 + i * 3
        col_z = 3 + i * 3
        
        body_color = colors[i % len(colors)]
        body_name = f'Тело {i + 1}'

        # 1. Рисуем траекторию (линию)
        fig.add_trace(go.Scatter3d(
            x=df[col_x], 
            y=df[col_y], 
            z=df[col_z],
            mode='lines',
            name=body_name,
            line=dict(color=body_color, width=4),
            opacity=0.8
        ))

        # 2. Рисуем начальную точку (маленькая точка)
        fig.add_trace(go.Scatter3d(
            x=[df[col_x].iloc[0]], 
            y=[df[col_y].iloc[0]], 
            z=[df[col_z].iloc[0]],
            mode='markers',
            name=f'Start {i+1}',
            marker=dict(size=3, color=body_color),
            showlegend=False
        ))

        # 3. Рисуем конечную точку (большая сфера)
        fig.add_trace(go.Scatter3d(
            x=[df[col_x].iloc[-1]], 
            y=[df[col_y].iloc[-1]], 
            z=[df[col_z].iloc[-1]],
            mode='markers',
            name=f'End {i+1}',
            marker=dict(size=6, color=body_color),
            showlegend=False
        ))

    # Настройка внешнего вида графика
    fig.update_layout(
        title="Траектории задачи N тел",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            # aspectmode='data' гарантирует, что оси будут иметь правильный масштаб 
            # (куб останется кубом, а не сплющится)
            aspectmode='data' 
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        legend=dict(x=0, y=1),
        template="plotly_dark" # Темная тема (космос)
    )

    print("Генерация графика...")
    fig.write_html(f"results/graphics/{filename}.html")
    fig.show()

if __name__ == "__main__":
    # Можно передать имя файла аргументом: python visualize.py output.csv
    # Иначе будет искать output.csv по умолчанию
    filename = 'output.csv'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    plot_trajectories(filename)