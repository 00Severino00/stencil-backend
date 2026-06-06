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

    # 1. Normalización estricta de tamaño para proteger la RAM de Render
    alto, ancho = img.shape[:2]
    max_dim = 1000
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Escala de grises y eliminación de ruido base
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_previo = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. EL SECRETO DEL ESTILÓGRAFO: Umbral Adaptativo de Gauss
    # En lugar de recortar a lo bruto, analiza el entorno de cada píxel.
    # El slider de tolerancia define el tamaño del bloque (grosor de la pluma)
    block_size = int(tolerance)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, min(49, block_size)) # Mantener el rango seguro

    # Genera las líneas ultra-limpias de Procreate
    stencil = cv2.adaptiveThreshold(
        blur_previo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, 4
    )

    # 4. Suavizado de bordes sub-píxel integrado (No consume RAM)
    # Suaviza los dientes de sierra sin destruir los detalles del trazo artístico
    resultado = cv2.medianBlur(stencil, 3)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    # Renderizado HD optimizado en memoria
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Filtro Bilateral para romper el pixelado remanente y definir las curvas
    limpio = cv2.bilateralFilter(img, 4, 75, 75)
    _, final_hd = cv2.threshold(limpio, 200, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
