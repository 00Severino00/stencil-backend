from fastapi import FastAPI, UploadFile, File
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
async def generate_stencil(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar un desenfoque Gaussiano fuerte para ignorar texturas pequeñas y ruidos del fondo
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Algoritmo Canny: Detecta los bordes principales e ignora los cambios suaves de luz
    # Esto creará líneas blancas sobre un fondo negro (muy similar a tu ejemplo de referencia)
    edges = cv2.Canny(blurred, 30, 100)
    
    # 4. Invertir la imagen para que las líneas queden negras y el fondo blanco (estilo calco tradicional)
    # Si prefieres que el fondo sea negro y las líneas blancas, borra la siguiente línea de código:
    stencil = cv2.bitwise_not(edges)
    
    _, encoded_img = cv2.imencode('.png', stencil)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
