import cv2
import os
import numpy as np
# Ruta de la carpeta de entrada y salida
input_folder = 'rostros_listos'
output_folder = 'bordes_detallados'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def preprocess_image(image):
    """
    Mejora el contraste de la imagen con CLAHE y aplica un filtro bilateral para suavizar.
    :param image: Imagen en escala de grises.
    :return: Imagen mejorada.
    """
    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Suavizar con un filtro bilateral
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    return smoothed


def auto_canny(image, sigma=0.33):
    """
    Ajusta automáticamente los umbrales de Canny en función de la mediana de la imagen.
    :param image: Imagen en escala de grises.
    :param sigma: Factor para ajustar el rango de umbrales.
    :return: Imagen con bordes detectados.
    """
    # Calcular la mediana de los píxeles de la imagen
    median = np.median(image)

    # Establecer umbrales en función del sigma
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # Aplicar Canny con los umbrales calculados
    edges = cv2.Canny(image, lower, upper)
    return edges

# Función para procesar y mejorar la imagen
def procesar_imagen(image_path, output_path):
    # Leer la imagen original
    img = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)
    edges = auto_canny(gray, 0.1)
    edges = cv2.dilate(edges, np.ones((5,5), dtype=np.int8))
    # Detectar los bordes con el algoritmo de Canny
    # edges = cv2.Canny(gray, 150, 200)
    
    # # Mejorar el contraste de la imagen original
    # # Usamos un filtro de aumento de contraste
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # l = cv2.equalizeHist(l)  # Ecualizar el canal L para mejorar el contraste
    # lab = cv2.merge((l, a, b))
    # img_contraste = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # # Combinar los bordes detectados con la imagen original
    # # Resaltamos los bordes, combinándolos con la imagen original en escala de grises
    # img_bnw = cv2.bitwise_and(img_contraste, img_contraste, mask=edges)
    
    # # Guardar la imagen resultante en la carpeta de salida
    cv2.imwrite(output_path, edges)


# Procesar todas las imágenes de la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)
        
        # Detectar los bordes y guardar la imagen con los bordes
        procesar_imagen(input_image_path, output_image_path)

print("Proceso completado.")