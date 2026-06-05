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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    stencil = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    _, encoded_img = cv2.imencode('.png', stencil)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 1. Aumentar el tamaño de la imagen al doble usando interpolación cúbica (Suaviza píxeles)
    alto, ancho = img.shape[:2]
    img_hd = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Aplicar un filtro morfológico para conectar líneas rotas y eliminar puntos de ruido aislados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Suaviza los bordes internos de los trazos negros
    img_hd = cv2.morphologyEx(img_hd, cv2.MORPH_CLOSE, kernel)

    # 3. Re-ajustar con un umbral estricto para que las curvas queden totalmente sólidas
    _, final_hd = cv2.threshold(img_hd, 127, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
