import tensorflow as tf
import os
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image

def load_dataset(bocetos_path, realistas_path, batch_size=32):
    bocetos = []
    realistas = []
    
    # Iterar en lotes
    for file_name in os.listdir(bocetos_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
            bocetos.append(load_image(os.path.join(bocetos_path, file_name)))
            realistas.append(load_image(os.path.join(realistas_path, file_name)))
        
        # Si hemos llegado al tamaño del batch, devolvemos las imágenes procesadas
        if len(bocetos) >= batch_size:
            yield np.array(bocetos), np.array(realistas)
            bocetos, realistas = [], []  # Limpiar para el siguiente lote

    # Asegurarse de que si quedan imágenes, se devuelvan
    if len(bocetos) > 0:
        yield np.array(bocetos), np.array(realistas)

# Llamar a la función de carga de datos en lotes
for bocetos_batch, realistas_batch in load_dataset('border_faces_boceto', 'faces_recortadas'):
    # Aquí puedes entrenar el modelo con el lote actual
    print(f"Cargado lote con {len(bocetos_batch)} imágenes")

# No necesitas hacer esta parte fuera del bucle, ya que `load_dataset` es un generador:
# bocetos, realistas = load_dataset('border_faces_boceto', 'faces_recortadas')  # Esto ya no es necesario
