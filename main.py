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

    # 1. Control estricto de resolución para no romper la memoria RAM de Render
    alto, ancho = img.shape[:2]
    max_dim = 1000
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * esca), int(alto * escala)), interpolation=cv2.INTER_AREA)
        alto, ancho = img.shape[:2]

    # 2. Convertir a escala de grises y validar fondo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray[0:20, 0:20]) < 127:
        gray = cv2.bitwise_not(gray)

    # 3. FILTRO DE LÍNEA PURA (Diferencia de Gaussianas)
    # En lugar de umbrales, restamos dos niveles de desenfoque. 
    # Esto elimina manchas de iluminación y deja SOLO las líneas del dibujo vivas.
    g1 = cv2.GaussianBlur(gray, (3, 3), 0)
    g2 = cv2.GaussianBlur(gray, (25, 25), 0)
    dog = cv2.divide(g1, g2, scale=255)

    # 4. Ajuste del grosor del estilógrafo mediante el slider
    # Mapeo óptimo para el control del trazo
    valor_tinta = max(200, min(250, 255 - int(tolerance)))
    _, binary = cv2.threshold(dog, valor_tinta, 255, cv2.THRESH_BINARY)

    # 5. SUPER-SMOOTHING (Efecto StreamLine de Procreate)
    # Duplicamos la escala virtualmente para redondear las esquinas pixeladas
    hd = cv2.resize(binary, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    hd_blur = cv2.GaussianBlur(hd, (3, 3), 0)
    _, hd_thresh = cv2.threshold(hd_blur, 220, 255, cv2.THRESH_BINARY)
    
    # Regresamos al tamaño final aplicando un antialiasing nativo super suave
    resultado = cv2.resize(hd_thresh, (ancho, alto), interpolation=cv2.INTER_AREA)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # El modo HD intensifica el negro del trazo y lima cualquier imperfección restante
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, final_hd = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
