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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(100)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Pasar a gris y aplicar un filtro para suavizar imperfecciones de fondo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Algoritmo Canny Puro para Line Art Estricto
    # La tolerancia define el umbral de detección de bordes
    # A menor tolerancia, entran más líneas. A mayor tolerancia, solo las líneas más fuertes.
    low_threshold = max(1, 150 - (tolerance * 7))
    high_threshold = max(50, 255 - (tolerance * 5))
    
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # 3. Invertir la imagen para que las líneas queden negras y el fondo blanco puro
    line_art = cv2.bitwise_not(edges)

    _, encoded_img = cv2.imencode('.png', line_art)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Escalado de alta calidad para vectorizar las líneas extraídas
    alto, ancho = img.shape[:2]
    img_hd = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    
    # Limpieza de píxeles sueltos para trazos continuos
    _, final_hd = cv2.threshold(img_hd, 200, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
