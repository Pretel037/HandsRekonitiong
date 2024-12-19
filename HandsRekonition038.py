import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Variables globales
contador_juntas = 0
contador_derecha_baja = 0
contador_quietas = 0
archivo_txt = "conteo_manos.txt"
posiciones_previas = []
tiempo_quieto = 2  # Segundos para considerar "quieto"
ultima_actualizacion = time.time()

def detectar_posicion(mano1, mano2):
    global contador_juntas, contador_derecha_baja, contador_quietas, ultima_actualizacion, posiciones_previas

    if mano1 and mano2:  # Ambas manos detectadas
        # Obtener coordenadas
        x1, y1 = mano1[0], mano1[1]  # Coordenadas mano izquierda
        x2, y2 = mano2[0], mano2[1]  # Coordenadas mano derecha

        # Verificar si las manos están juntas (distancia pequeña)
        distancia = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distancia < 100:  # Umbral para considerar las manos juntas
            contador_juntas += 1
            guardar_en_txt("Puños Juntos")
            return "Puños Juntos"

        # Verificar si las manos se están trasladando juntas
        if posiciones_previas:
            x1_prev, y1_prev = posiciones_previas[0][0]
            x2_prev, y2_prev = posiciones_previas[0][1]
            desplazamiento_izquierda = ((x1 - x1_prev) ** 2 + (y1 - y1_prev) ** 2) ** 0.5
            desplazamiento_derecha = ((x2 - x2_prev) ** 2 + (y2 - y2_prev) ** 2) ** 0.5

            if desplazamiento_izquierda > 20 and desplazamiento_derecha > 20:  # Ajustar el umbral
                guardar_en_txt("Movimiento de Traslado")
                return "Movimiento de Traslado"

        # Verificar si la mano derecha está moviéndose hacia afuera
        if x2 > x1 + 100:  # Umbral para movimiento lateral
            guardar_en_txt("Mano Derecha Moviendo")
            return "Mano Derecha Moviendo"

    # Actualizar posiciones previas para el cálculo de movimientos
    posiciones_previas = [(mano1, mano2)]
    return ""


def guardar_en_txt(evento):
    with open(archivo_txt, "a") as file:
        file.write(f"{evento}\n")

# Funciones de video y GUI
def procesar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    
    def actualizar_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        # Redimensionar el frame para que ocupe menos espacio
        frame = cv2.resize(frame, (600, 800))  # Ajusta las dimensiones según sea necesario

        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)
        estado = ""

        # Procesar detecciones
        if resultados.multi_hand_landmarks:
            manos = []
            for hand_landmarks in resultados.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                manos.append((x, y))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar posiciones
            if len(manos) == 2:
                estado = detectar_posicion(manos[0], manos[1])

        # Mostrar contador y estado
        frame = cv2.putText(frame, f"Juntas: {contador_juntas}  Derecha Baja: {contador_derecha_baja}  Quietas: {contador_quietas}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"Estado: {estado}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convertir frame a imagen Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.configure(image=imgtk)
        panel.after(10, actualizar_frame)


    # Iniciar el loop de video
    actualizar_frame()

def cargar_video():
    ruta_video = filedialog.askopenfilename(title="Seleccionar video", filetypes=[("Video files", "*.mp4;*.avi")])
    if ruta_video:
        procesar_video(ruta_video)

# GUI con Tkinter
root = tk.Tk()
root.title("Reconocedor de Posiciones de Manos")

# Botones y panel
btn_cargar = tk.Button(root, text="Cargar Video", command=cargar_video)
btn_cargar.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()
