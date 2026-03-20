from PIL import Image

def move_points(poly: dict, white_frame: tuple[int, int, int, int]):
    left, upper, _, _ = white_frame
    for key, value in poly.items():
        if key.startswith('x'):
            value = float(value) - left
        elif key.startswith('y'):
            value = float(value) - upper
        poly[key] = value
    return poly

def cut_white_frame(img: Image.Image, white_frame: tuple[int, int, int, int]) -> Image.Image:
    left, upper, right, lower = white_frame
    cropped_img = img.crop((left, upper, img.size[0] - right, img.size[1] - lower))
    return cropped_img