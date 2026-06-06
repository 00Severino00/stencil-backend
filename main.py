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

    # 1. Pasar a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Detección de fondo para evitar plastas negras
    muestra_fondo = gray[0:20, 0:20]
    if np.mean(muestra_fondo) < 127:
        gray = cv2.bitwise_not(gray)

    # 3. Super-Sampling controlado (Escalado x2 de alta fidelidad)
    # Creamos píxeles intermedios virtuales para calcular curvas suaves
    alto, ancho = gray.shape[:2]
    img_alta = cv2.resize(gray, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)

    # 4. Limpieza del ruido e imperfecciones de capturas
    filtrada = cv2.bilateralFilter(img_alta, 5, 50, 50)

    # 5. EL SECRETO DEL ESTILÓGRAFO: Campo de distancia suavizado (Anti-aliasing de Procreate)
    # En lugar de cortar el píxel bruscamente, usamos un umbral blando con curvas de Gauss
    # El slider controla el grosor exacto del estilógrafo (tinta más densa o fina)
    valor_umbral = max(40, min(240, 255 - (int(tolerance) * 3)))
    
    # Creamos una máscara de suavizado sub-píxel
    _, mascara_binaria = cv2.threshold(filtrada, valor_umbral, 255, cv2.THRESH_BINARY)
    
    # Aplicamos un desenfoque sutil y reducimos de tamaño con interpolación de área 
    # Esto fusiona los "dientes de sierra" en curvas continuas, líquidas y perfectas
    lineas_suaves = cv2.GaussianBlur(mascara_binaria, (3, 3), 0)
    resultado_final = cv2.resize(lineas_suaves, (ancho, alto), interpolation=cv2.INTER_AREA)

    _, encoded_img = cv2.imencode('.png', resultado_final)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # El botón HD refina aún más las curvas simulando un trazo vectorial pulido
    alto, ancho = img.shape[:2]
    hd_upscale = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    
    hd_blur = cv2.GaussianBlur(hd_upscale, (3, 3), 0)
    _, hd_final = cv2.threshold(hd_blur, 210, 255, cv2.THRESH_BINARY)
    
    resultado_hd = cv2.resize(hd_final, (ancho, alto), interpolation=cv2.INTER_AREA)

    _, encoded_img = cv2.imencode('.png', resultado_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
