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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(10)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Pasar a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Suavizado bilateral para mantener los bordes duros pero eliminar texturas de piel/ruido
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Mapear el Line Art Real mediante una estructura de Umbral Adaptativo dinámico.
    # Usamos la tolerancia que viene desde la interfaz web.
    # El tamaño del bloque determina el grosor y el valor de tolerancia filtra las líneas falsas.
    block_size = 11
    constant_val = tolerance # Este valor viene del slider de la web (típicamente entre 1 y 20)
    
    line_art = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, constant_val
    )

    _, encoded_img = cv2.imencode('.png', line_art)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Escalado avanzado para vectorizar y limpiar los bordes del Line Art
    alto, ancho = img.shape[:2]
    img_hd = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    
    # Filtro de limpieza para trazos continuos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_hd = cv2.morphologyEx(img_hd, cv2.MORPH_CLOSE, kernel)
    
    _, final_hd = cv2.threshold(img_hd, 127, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
