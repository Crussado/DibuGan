import cv2
import numpy as np
import random
import os

def fisheye_effect(image, strength=0.2):
    """
    Aplica un efecto de ojo de pez suave a la imagen.
    :param image: Imagen de entrada.
    :param strength: Intensidad del efecto, valores menores hacen el efecto más sutil.
    :return: Imagen con efecto de ojo de pez.
    """
    # Dimensiones de la imagen
    height, width = image.shape[:2]

    # Crear una malla de coordenadas normalizada
    K = min(height, width) / 2  # Constante para el cálculo
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Aplicar distorsión de ojo de pez suave
    distortion = 1 + strength * (R**2)
    X_distorted = X / distortion
    Y_distorted = Y / distortion

    # Remapear la imagen con las coordenadas distorsionadas
    map_x = ((X_distorted + 1) * width - 1) / 2
    map_y = ((Y_distorted + 1) * height - 1) / 2

    fisheye_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return fisheye_image

def apply_random_transformations(edge_image, intensity=0.5):
    """
    Aplica una serie de transformaciones aleatorias a una imagen de bordes.
    :param edge_image: Imagen binaria con bordes (0 y 255).
    :param intensity: Nivel de intensidad de las transformaciones (0.1 a 1.0).
    :return: Imagen con efectos aplicados.
    """
    h, w, _ = edge_image.shape

    # Crear una copia de la imagen para modificar
    transformed_image = edge_image.copy()

    # Efecto 1: Pequeñas rotaciones
    angle = random.uniform(-15, 15) * intensity  # Rotación pequeña aleatoria
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    transformed_image = cv2.warpAffine(transformed_image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Efecto 2: Escalado no uniforme
    scale_x = 1 + random.uniform(-0.2, 0.2) * intensity
    scale_y = 1 + random.uniform(-0.2, 0.2) * intensity
    matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    transformed_image = cv2.warpAffine(transformed_image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return transformed_image

folder = './bordes_detallados'
for file in os.listdir(folder):
    image_path = os.path.join(folder, file)
    img = cv2.imread(image_path)
    img = apply_random_transformations(img)
    cv2.imwrite(image_path, img)

