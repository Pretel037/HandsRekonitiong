import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from collections import deque

# Configuración de Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables globales para contador
contador_juntas = 0
contador_derecha_baja = 0
contador_quietas = 0

# Función para guardar el conteo en un archivo de texto
def guardar_conteo():
    with open("conteo_manos.txt", "w") as file:
        file.write(f"Manos juntas: {contador_juntas}\n")
        file.write(f"Mano derecha baja: {contador_derecha_baja}\n")
        file.write(f"Manos quietas: {contador_quietas}\n")
    print("Conteo guardado en conteo_manos.txt")

# Función principal para procesar video
def procesar_video(ruta_video):
    global contador_juntas, contador_derecha_baja, contador_quietas
    
    # Cargar el video
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    last_positions = deque(maxlen=5)  # Almacena últimas posiciones para detectar "manos quietas"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar el video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Detectar manos y calcular posiciones
        if results.multi_hand_landmarks:
            x_positions = []
            y_positions = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar puntos de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Obtener coordenadas de la muñeca (punto 0)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                x_positions.append(wrist.x)
                y_positions.append(wrist.y)
            
            # Comprobar posiciones
            if len(x_positions) == 2:  # Si hay 2 manos
                if abs(x_positions[0] - x_positions[1]) < 0.05:  # Manos juntas
                    contador_juntas += 1
                    print("Manos juntas detectadas")

            elif len(x_positions) == 1:  # Si hay solo una mano
                if y_positions[0] > 0.8:  # Mano derecha baja (condición arbitraria)
                    contador_derecha_baja += 1
                    print("Mano derecha baja detectada")
            
            # Agregar última posición
            last_positions.append((x_positions, y_positions))
        else:
            # Si no hay detección, verificar si se quedaron quietas
            if len(last_positions) == 5 and all(len(pos[0]) == 0 for pos in last_positions):
                contador_quietas += 1
                print("Manos quietas detectadas")
        
        # Mostrar el video
        cv2.imshow("Reconocedor de manos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar el conteo al finalizar
    guardar_conteo()
    cap.release()
    cv2.destroyAllWindows()

# Interfaz gráfica con Tkinter
def seleccionar_video():
    ruta_video = filedialog.askopenfilename(title="Seleccionar Video", filetypes=[("Archivos de video", "*.mp4;*.avi")])
    if ruta_video:
        procesar_video(ruta_video)

# Configuración de la ventana Tkinter
ventana = tk.Tk()
ventana.title("Reconocedor de Posición de Manos")
ventana.geometry("400x200")

# Botón para seleccionar video
btn_seleccionar = tk.Button(ventana, text="Seleccionar Video", command=seleccionar_video, bg="lightblue", fg="black", font=("Arial", 12))
btn_seleccionar.pack(pady=20)

# Etiqueta de instrucciones
label_instruccion = tk.Label(ventana, text="Presiona 'q' para salir durante el procesamiento del video.", font=("Arial", 10))
label_instruccion.pack(pady=10)

# Iniciar la interfaz
ventana.mainloop()
