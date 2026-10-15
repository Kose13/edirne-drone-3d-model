#!/usr/bin/env python3
"""
Създаване на GIF анимация от гледна точка на дрон DJI Mavic Mini 2
над град Одрин от височина 120m, поглед към центъра, 16:9 формат, 8 секунди.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os

# Параметри за анимацията
DURATION_SECONDS = 8
FPS = 24
TOTAL_FRAMES = DURATION_SECONDS * FPS
HEIGHT = 120  # височина на дрона в метри
ALTITUDE = HEIGHT  # височина над земята

# Размери на изходния GIF (16:9)
WIDTH_PX = 1280  # 16:9 - 1280x720
HEIGHT_PX = 720

# Координати на Одрин (приблизителни за визуализация)
# Център на Одрин: 41.6756° N, 26.4833° E
CENTER_X = 0
CENTER_Y = 0

# Параметри на дрона
DRONE_SPEED = 5.0  # m/s (приблизително)
INITIAL_DISTANCE = 200  # начално разстояние от центъра
FINAL_DISTANCE = 50  # крайно разстояние от центъра

# Цветове
SKY_COLOR = (135, 206, 235)  # sky blue
GROUND_COLOR = (34, 139, 34)  # forest green
BUILDING_COLOR = (169, 169, 169)  # dark gray
ROAD_COLOR = (110, 110, 110)  # gray
RIVER_COLOR = (65, 105, 225)  # royal blue

def create_city_layout():
    """Създава процедурална карта на Одрин с улици, сгради и река"""
    city = {}
    
    # Река Марица (приблизително направление през Одрин)
    city['river'] = [
        np.array([-150, -200, 0]),
        np.array([-100, -150, 0]),
        np.array([-50, -100, 0]),
        np.array([0, -50, 0]),
        np.array([50, 0, 0]),
        np.array([100, 50, 0]),
        np.array([150, 100, 0]),
    ]
    
    # Улици (основни булеварди)
    city['roads'] = [
        # Хоризонтални улици
        np.array([[-200, -100, 0], [200, -100, 0]]),
        np.array([[-200, -50, 0], [200, -50, 0]]),
        np.array([[-200, 0, 0], [200, 0, 0]]),
        np.array([[-200, 50, 0], [200, 50, 0]]),
        np.array([[-200, 100, 0], [200, 100, 0]]),
        # Вертикални улици
        np.array([[-100, -200, 0], [-100, 200, 0]]),
        np.array([[-50, -200, 0], [-50, 200, 0]]),
        np.array([[0, -200, 0], [0, 200, 0]]),
        np.array([[50, -200, 0], [50, 200, 0]]),
        np.array([[100, -200, 0], [100, 200, 0]]),
    ]
    
    # Сгради (групи от сгради)
    city['buildings'] = []
    
    # Централна част - по-големи сгради
    for x in range(-80, 90, 20):
        for y in range(-80, 90, 20):
            if abs(x) < 20 and abs(y) < 20:
                height = np.random.randint(30, 80)
            else:
                height = np.random.randint(5, 40)
            width = np.random.randint(10, 30)
            depth = np.random.randint(10, 30)
            city['buildings'].append({
                'position': np.array([x, y, 0]),
                'width': width,
                'depth': depth,
                'height': height,
                'color': BUILDING_COLOR
            })
    
    # Пътища (кръжок около центъра)
    city['circular_road'] = []
    for angle in np.linspace(0, 2*np.pi, 100):
        radius = 120
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        city['circular_road'].append(np.array([x, y, 0]))
    
    return city

def project_3d_to_2d(points_3d, camera_pos, camera_target, fov=60, aspect_ratio=16/9):
    """
    Проектира 3D точки върху 2D равнина (камера гледна точка)
    """
    # Камера параметри
    camera_dir = camera_target - camera_pos
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    
    # Вектор нагоре (Z-axis)
    up = np.array([0, 0, 1])
    
    # Вектор на дясно (X-axis)
    right = np.cross(up, camera_dir)
    right = right / np.linalg.norm(right)
    
    # Вектор нагоре в камера координати
    camera_up = np.cross(camera_dir, right)
    
    # Проекция
    projected = []
    for point in points_3d:
        # Вектор от камерата към точката
        vec = point - camera_pos
        
        # Разстояние по посока на камерата
        depth = np.dot(vec, camera_dir)
        
        if depth <= 0:
            # Точката е зад камерата
            projected.append(None)
            continue
        
        # Проекция върху равнината на камерата
        right_component = np.dot(vec, right)
        up_component = np.dot(vec, camera_up)
        
        # Фокусно разстояние
        focal_length = 1.0 / np.tan(np.radians(fov / 2))
        
        # Нормализирани координати
        x_norm = right_component * focal_length / depth
        y_norm = up_component * focal_length / depth
        
        # Преобразуване в пиксели
        x_pix = (x_norm * aspect_ratio + 1) * WIDTH_PX / 2
        y_pix = (1 - y_norm) * HEIGHT_PX / 2
        
        projected.append((x_pix, y_pix, depth))
    
    return projected

def draw_frame(frame_idx, city, camera_path):
    """Рисува един кадър от анимацията"""
    img = Image.new('RGB', (WIDTH_PX, HEIGHT_PX), SKY_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Текуща позиция на камерата (дрон)
    t = frame_idx / TOTAL_FRAMES
    camera_pos = camera_path(frame_idx)
    camera_target = np.array([CENTER_X, CENTER_Y, 0])  # Гледа към центъра
    
    # Проектираме границите на терена
    ground_corners = [
        np.array([-300, -300, 0]),
        np.array([300, -300, 0]),
        np.array([300, 300, 0]),
        np.array([-300, 300, 0]),
    ]
    
    projected_ground = project_3d_to_2d(ground_corners, camera_pos, camera_target)
    
    # Рисуваме небето (вече е зададено като фон)
    
    # Рисуваме терена
    ground_points = []
    for point in projected_ground:
        if point is not None:
            ground_points.append((point[0], point[1]))
    
    if len(ground_points) >= 3:
        draw.polygon(ground_points, fill=GROUND_COLOR)
    
    # Рисуваме реката
    river_points = []
    for point in city['river']:
        proj = project_3d_to_2d([point], camera_pos, camera_target)[0]
        if proj is not None:
            river_points.append((proj[0], proj[1]))
    
    if len(river_points) >= 2:
        draw.line(river_points, fill=RIVER_COLOR, width=8)
    
    # Рисуваме пътищата
    for road in city['roads']:
        road_points = []
        for point in road:
            proj = project_3d_to_2d([point], camera_pos, camera_target)[0]
            if proj is not None:
                road_points.append((proj[0], proj[1]))
        
        if len(road_points) >= 2:
            draw.line(road_points, fill=ROAD_COLOR, width=4)
    
    # Рисуваме кръглото шосе
    circular_points = []
    for point in city['circular_road']:
        proj = project_3d_to_2d([point], camera_pos, camera_target)[0]
        if proj is not None:
            circular_points.append((proj[0], proj[1]))
    
    if len(circular_points) >= 2:
        draw.line(circular_points, fill=ROAD_COLOR, width=3)
    
    # Рисуваме сградите (от най-далеч до най-близо, за да работи z-buffering)
    # Сортираме по разстояние от камерата
    buildings_with_depth = []
    for building in city['buildings']:
        center_proj = project_3d_to_2d([building['position']], camera_pos, camera_target)[0]
        if center_proj is not None:
            buildings_with_depth.append((building, center_proj[2]))
    
    # Сортираме по обратна дълбочина (далечните се рисуват първи)
    buildings_with_depth.sort(key=lambda x: x[1], reverse=True)
    
    for building, depth in buildings_with_depth:
        # Проектираме 4-те ъгъла на сградата
        half_w = building['width'] / 2
        half_d = building['depth'] / 2
        
        corners_3d = [
            building['position'] + np.array([-half_w, -half_d, 0]),
            building['position'] + np.array([half_w, -half_d, 0]),
            building['position'] + np.array([half_w, half_d, 0]),
            building['position'] + np.array([-half_w, half_d, 0]),
        ]
        
        # Проектираме горната част на сградата
        top_corners_3d = [
            building['position'] + np.array([-half_w, -half_d, building['height']]),
            building['position'] + np.array([half_w, -half_d, building['height']]),
            building['position'] + np.array([half_w, half_d, building['height']]),
            building['position'] + np.array([-half_w, half_d, building['height']]),
        ]
        
        # Проекция на долната и горната част
        bottom_proj = project_3d_to_2d(corners_3d, camera_pos, camera_target)
        top_proj = project_3d_to_2d(top_corners_3d, camera_pos, camera_target)
        
        # Филтрираме валидните точки
        bottom_valid = [p for p in bottom_proj if p is not None]
        top_valid = [p for p in top_proj if p is not None]
        
        if len(bottom_valid) >= 3 and len(top_valid) >= 3:
            # Рисуваме стените
            for i in range(len(bottom_valid)):
                j = (i + 1) % len(bottom_valid)
                if i < len(top_valid) and j < len(top_valid):
                    # Долна линия
                    draw.line([bottom_valid[i][:2], bottom_valid[j][:2]], 
                             fill=tuple(c - 20 for c in building['color']), width=1)
                    # Горна линия
                    draw.line([top_valid[i][:2], top_valid[j][:2]], 
                             fill=tuple(c - 20 for c in building['color']), width=1)
                    # Вертикална линия
                    draw.line([bottom_valid[i][:2], top_valid[i][:2]], 
                             fill=tuple(c - 40 for c in building['color']), width=1)
            
            # Рисуваме покрива
            draw.polygon([p[:2] for p in top_valid], fill=building['color'])
    
    # Добавяме надпис
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # Текст в горния ляв ъгъл
    draw.text((20, 20), f"DJI Mavic Mini 2 - {HEIGHT}m", fill=(255, 255, 255), font=font)
    draw.text((20, 70), "Одрин, България", fill=(255, 255, 255), font=font)
    
    # Прогрес бар
    progress = int((frame_idx / TOTAL_FRAMES) * 100)
    draw.rectangle([(20, HEIGHT_PX - 50), (WIDTH_PX - 20, HEIGHT_PX - 30)], 
                   fill=(50, 50, 50), outline=(200, 200, 200))
    draw.rectangle([(20, HEIGHT_PX - 50), (20 + (WIDTH_PX - 40) * progress / 100, HEIGHT_PX - 30)], 
                   fill=(0, 200, 0), outline=(200, 200, 200))
    
    return img

def camera_movement_path(frame_idx):
    """
    Дефинира пътя на камерата (дрон)
    Дронът започва отдалеч и се приближава към центъра
    """
    t = frame_idx / TOTAL_FRAMES
    
    # Начална позиция: отдалеч на югозапад
    start_x = INITIAL_DISTANCE
    start_y = INITIAL_DISTANCE
    start_z = ALTITUDE
    
    # Крайна позиция: по-близо до центъра
    end_x = FINAL_DISTANCE * np.cos(np.pi / 4)
    end_y = FINAL_DISTANCE * np.sin(np.pi / 4)
    end_z = ALTITUDE
    
    # Интерполация
    x = start_x + (end_x - start_x) * t
    y = start_y + (end_y - start_y) * t
    z = start_z + (end_z - start_z) * t
    
    # Добавяме леко люлеене за по-реалистичен ефект
    sway_x = 5 * np.sin(t * np.pi * 2)
    sway_y = 5 * np.cos(t * np.pi * 1.5)
    
    return np.array([x + sway_x, y + sway_y, z])

def create_animation():
    """Създава анимацията и я записва като GIF"""
    print("Създаване на карта на Одрин...")
    city = create_city_layout()
    
    print(f"Генериране на {TOTAL_FRAMES} кадра...")
    
    # Създаваме директория за кадрите
    frames_dir = "/tmp/odrin_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Генерираме всички кадри
    frames = []
    for i in range(TOTAL_FRAMES):
        frame = draw_frame(i, city, camera_movement_path)
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame.save(frame_path)
        frames.append(frame)
        
        if (i + 1) % 24 == 0:
            print(f"  Готово: {i + 1}/{TOTAL_FRAMES} кадра ({((i + 1) / TOTAL_FRAMES * 100):.1f}%)")
    
    print("\nСъединяване на кадрите в GIF...")
    
    # Записваме като GIF
    output_path = "/workspace/odrin_drone_animation.gif"
    
    # Използваме Pillow за създаване на GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,  # милисекунди на кадър
        loop=0,  # безкраен loop
        optimize=True,
        quality=85
    )
    
    print(f"\n✓ Анимацията е създадена: {output_path}")
    print(f"  Продължителност: {DURATION_SECONDS} секунди")
    print(f"  Размери: {WIDTH_PX}x{HEIGHT_PX} (16:9)")
    print(f"  Фреймове: {TOTAL_FRAMES} ({FPS} fps)")
    print(f"  Височина на дрон: {HEIGHT}m")
    
    # Почистване на временните файлове
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    
    return output_path

if __name__ == "__main__":
    create_animation()
