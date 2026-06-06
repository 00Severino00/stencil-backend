from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-stencil/")
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(2)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Blindaje de Memoria: Evita que Render colapse con imágenes pesadas de iPad
    alto, ancho = img.shape[:2]
    max_dim = 1200
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Aislamiento de frecuencias y eliminación de ruido de compresión de pantallas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # 3. Extracción de la máscara base del trazo
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # 4. MOTOR VECTORIAL (Efecto Estilógrafo de Procreate)
    # Buscamos las líneas maestras y descartamos el ruido microscópico del fondo
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Creamos un lienzo nuevo, blanco y puro (Cero manchas de fondo)
    canvas = np.ones_like(gray) * 255

    # Configuración del grosor del estilógrafo basado en el Slider de tolerancia
    # tolerance vendrá del frontend en un rango de 1 a 4
    grosor_linea = int(tolerance)

    for c in contours:
        # Filtro de escala: Ignora motas de polvo o artefactos menores a 3 píxeles
        if cv2.contourArea(c) > 3:
            # LINE_AA genera un suavizado sub-píxel idéntico al trazo digital continuo
            cv2.drawContours(canvas, [c], -1, (0, 0, 0), thickness=grosor_linea, lineType=cv2.LINE_AA)

    _, encoded_img = cv2.imencode('.png', canvas)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
