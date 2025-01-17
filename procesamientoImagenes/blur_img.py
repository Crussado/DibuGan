import cv2
import mediapipe as mp
import numpy as np
import os
# Inicializar el detector de rostros de MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def apply_blur_to_background(image_path, blur_intensity=25):
    """
    Aplica un desenfoque al fondo de una imagen manteniendo el rostro nítido.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Usar MediaPipe para detectar el rostro
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if not results.detections:
            print("No se detectó ningún rostro en la imagen.")
            return image

        # Crear una máscara para el fondo
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for detection in results.detections:
            # Obtener el bounding box del rostro
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Dibujar un rectángulo en la máscara para el rostro
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # Invertir la máscara para el fondo
        background_mask = cv2.bitwise_not(mask)

        # Aplicar blur al fondo
        blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
        blurred_background = cv2.bitwise_and(blurred_image, blurred_image, mask=background_mask)

        # Combinar el rostro original con el fondo desenfocado
        face_region = cv2.bitwise_and(image, image, mask=mask)
        final_image = cv2.add(face_region, blurred_background)

        return final_image

INPATH = "./rostros"
OUTPATH = "./rostros_listos"
imgs = os.listdir(INPATH)

for img in imgs:

    # Procesar la imagen
    processed_image = apply_blur_to_background(INPATH + '/' + img, blur_intensity=25)

    # Guardar y mostrar el resultado
    cv2.imwrite(OUTPATH + '/' + img, processed_image)
    # cv2.imshow("Imagen Procesada", processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
