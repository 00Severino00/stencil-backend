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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(15)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Escala inteligente: Redimensionar solo si la imagen es masiva para proteger la RAM de Render
    alto, ancho = img.shape[:2]
    max_dim = 1200
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. DETECCIÓN AUTOMÁTICA DE FONDO (Inteligencia anti-plastas)
    # Calculamos la esquina superior izquierda para saber si predomina el negro o el blanco
    muestra_fondo = gray[0:20, 0:20]
    es_fondo_oscuro = np.mean(muestra_fondo) < 127
    
    if es_fondo_oscuro:
        # Si la imagen es estilo Medusa (Fondo negro, líneas claras), la invertimos para trabajar en limpio
        gray = cv2.bitwise_not(gray)

    # 4. Umbralizado Adaptativo Gaussiano (Vuelve a la precisión que te gustó)
    # Ajustamos dinámicamente el tamaño del bloque según el slider de la interfaz
    block_size = max(3, int(tolerance) * 2 + 1)
    if block_size % 2 == 0:
        block_size += 1

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, 9
    )

    # 5. Estilógrafo y Limpieza de bordes pixelados
    # Aplicamos un desenfoque sutil y re-umbralizamos para fusionar píxeles rotos en curvas fluidas
    suave = cv2.GaussianBlur(thresh, (3, 3), 0)
    _, resultado = cv2.threshold(suave, 180, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Render HD libre de caídas de RAM: Suaviza imperfecciones del trazo sin inflar la imagen
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, final_hd = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
