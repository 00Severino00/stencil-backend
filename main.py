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

    # 1. Escala de grises y súper suavizado para eliminar el ruido granulado del fondo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. CAPA 1: Bordes y líneas maestras ultra nítidas (Estilo contorno puro)
    edges = cv2.Canny(blurred, 30, 90)
    edges_inv = cv2.bitwise_not(edges)
    
    # 3. CAPA 2: Sombras vectorizadas por umbrales (Para que no quede borroso ni tape el fondo)
    # Creamos un mapa donde dividimos los tonos en pasos limpios
    stencil_multilevel = np.zeros_like(blurred)
    
    # Si el píxel es muy oscuro -> Negro puro (Línea/Relleno profundo)
    stencil_multilevel[blurred < 65] = 0
    # Si es sombra media -> Gris de transferencia para guiar agujas de sombreado
    stencil_multilevel[(blurred >= 65) & (blurred < 140)] = 110
    # Si es zona clara o fondo -> Blanco absoluto (Limpio para transfer)
    stencil_multilevel[blurred >= 140] = 255
    
    # 4. Fusionar contornos duros con las sombras estilizadas
    final_stencil = cv2.min(stencil_multilevel, edges_inv)
    
    _, encoded_img = cv2.imencode('.png', final_stencil)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
