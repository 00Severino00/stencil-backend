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

    # 1. Pasar a gris y aplicar un filtro para suavizar imperfecciones de la piel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 2. CAPA DE SOMBRAS: Crear un efecto de sombreado suave (estilo carboncillo)
    gray_inv = cv2.bitwise_not(smooth)
    # Desenfoque profundo para capturar volúmenes de luz
    blur_inv = cv2.GaussianBlur(gray_inv, (21, 21), 0)
    # Mezclar para extraer sombras intermedias en escala de grises
    shadows = cv2.divide(smooth, cv2.bitwise_not(blur_inv), scale=256)
    
    # 3. CAPA DE LÍNEAS: Detección nítida de bordes maestros
    edges = cv2.Canny(smooth, 25, 85)
    edges_inv = cv2.bitwise_not(edges)
    
    # 4. FUSIÓN INTELIGENTE: Multiplicar las líneas duras con las sombras suaves
    # Esto te dará el contorno marcado de la Medusa con sus degradés internos
    result = cv2.multiply(shadows, edges_inv, scale=1/255)
    
    # Convertir a imagen PNG final
    _, encoded_img = cv2.imencode('.png', result)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
