import math

def extract_points(poly: dict) -> list[tuple, tuple, tuple, tuple]:
    points = []
    for i in range(1, 5):
        x = float(poly[f'x{i}'])
        y = float(poly[f'y{i}'])
        points.append((x, y))
    return points

def absolute2relative(size: tuple[int, int], poly):
    width, height = size
    for key, value in poly.items():
        if key.startswith('x'):
            value = float(value) / width
        elif key.startswith('y'):
            value = float(value) / height
        poly[key] = value
    return poly

def dota_polygon_to_yolo_obb(poly: dict, img_w, img_h):
    """
    poly: dict, e.g.
        {
          'x1': '486', 'x2': '507', 'x3': '500', 'x4': '477',
          'y1': '567', 'y2': '563', 'y3': '512', 'y4': '516'
        }
    img_w, img_h: 原图宽高
    return: (cx, cy, w, h, angle_deg)  # 全部为 YOLO OBB 格式（cx,cy,w,h 归一化，角度为度）
    """

    # 1. 取出四个点（转 float）
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = extract_points(poly)

    # 2. 中心点（四点平均）
    cx = (x1 + x2 + x3 + x4) / 4.0
    cy = (y1 + y2 + y3 + y4) / 4.0

    # 3. 定义宽 w 为 p1->p2 的长度，高 h 为 p2->p3 的长度
    w = math.hypot(x2 - x1, y2 - y1)
    h = math.hypot(x3 - x2, y3 - y2)

    # 4. 角度：p1->p2 的方向角（以 x 轴为基准）
    angle_rad = math.atan2(y2 - y1, x2 - x1)  # [-pi, pi]
    angle_deg = math.degrees(angle_rad)       # 转成度

    # 可选：把角度规范到 [0, 180) 或 [-90, 90) 视你的训练代码而定
    # 这里示例规范到 [-90, 90)
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg >= 90:
        angle_deg -= 180

    # 5. 归一化到 YOLO 格式
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w / img_w
    h_n = h / img_h

    return cx_n, cy_n, w_n, h_n, angle_deg
