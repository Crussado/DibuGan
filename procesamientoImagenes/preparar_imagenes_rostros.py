import cv2
import os

def renombrar_imagenes(carpeta):
    # Obtener todos los archivos en la carpeta
    files = os.listdir(carpeta)

    # Filtrar solo los archivos con extensión .jpg, .jpeg, .png
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]

    # Renombrar los archivos
    for i, file in enumerate(image_files, start=1):
        # Obtener la extensión del archivo
        file_extension = os.path.splitext(file)[1]
        
        # Construir el nuevo nombre
        new_name = f"{i}{file_extension}"
        
        # Ruta completa para el archivo actual y el nuevo nombre
        old_file_path = os.path.join(carpeta, file)
        new_file_path = os.path.join(carpeta, new_name)
        
        # Renombrar el archivo
        os.rename(old_file_path, new_file_path)

        print(f"Renombrado: {file} -> {new_name}")


def procesar_rostros(input_folder, output_folder, cascade_path="haarcascade_frontalface_default.xml", margin_percent=0.2, resize_width=256, resize_height=256):
    # Asegúrate de que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Cargar el clasificador Haar para detección de rostros
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Función para expandir el área del recorte con márgenes
    def expand_roi(x, y, w, h, margin, img_width, img_height):
        x_new = max(x - margin, 0)
        y_new = max(y - margin, 0)
        w_new = min(w + 2 * margin, img_width - x_new)
        h_new = min(h + 2 * margin, img_height - y_new)
        return x_new, y_new, w_new, h_new

    # Procesar cada imagen en la carpeta
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Leer la imagen
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"No se pudo leer la imagen: {filename}")
                continue

            # Obtener dimensiones de la imagen
            img_height, img_width = image.shape[:2]

            # Convertir la imagen a escala de grises (necesario para el clasificador Haar)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detectar rostros en la imagen
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Usar el primer rostro detectado
                x, y, w, h = faces[0]

                # Expander el área del recorte
                margin = int(margin_percent * h)  # margen basado en el porcentaje del alto del rostro
                x, y, w, h = expand_roi(x, y, w, h, margin, img_width, img_height)

                # Recortar la región ampliada
                face_img = image[y:y + h, x:x + w]

                # Redimensionar a las dimensiones deseadas
                face_img_resized = cv2.resize(face_img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

                # Guardar la imagen procesada
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, face_img_resized)

                print(f"Procesada: {filename}")
            else:
                print(f"No se detectaron rostros en: {filename}")


input_folder = "f"
output_folder = "frec"
procesar_rostros(input_folder, output_folder)

renombrar_imagenes('faces_recortadas')
