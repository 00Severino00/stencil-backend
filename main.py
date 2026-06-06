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

    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Súper-muestreo (Upscaling) para trabajar a nivel sub-píxel y evitar el pixelado
    alto, ancho = gray.shape[:2]
    img_alta = cv2.resize(gray, (ancho * 3, alto * 3), interpolation=cv2.INTER_CUBIC)
    
    # 3. Eliminar suciedad de fondo e imperfecciones con un filtro Bilateral
    # Esto mantiene los bordes afilados pero borra el ruido/manchas del fondo
    filtrada = cv2.bilateralFilter(img_alta, 9, 75, 75)
    
    # 4. Umbralizado Adaptativo Avanzado optimizado para texto/líneas finas de capturas
    # Adaptamos el bloque de búsqueda según el slider (valores entre 2 y 25)
    block_size = max(3, int(tolerance) * 2 + 1)
    if block_size % 2 == 0:
        block_size += 1
        
    thresh = cv2.adaptiveThreshold(
        filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, 7
    )
    
    # 5. Suavizado de curvas (Efecto Estilógrafo / Estabilizador de Procreate)
    # Mediante un desenfoque sutil y un re-umbralizado, obligamos a los píxeles 
    # escalonados ("dientes de sierra") a fusionarse en curvas continuas y limpias.
    blended = cv2.GaussianBlur(thresh, (3, 3), 0)
    _, line_art_final = cv2.threshold(blended, 180, 255, cv2.THRESH_BINARY)

    # Volvemos a su tamaño original para optimizar la descarga sin perder el suavizado
    resultado = cv2.resize(line_art_final, (ancho, alto), interpolation=cv2.INTER_AREA)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # El Render HD aplica una interpolación de vectores simulada (Supersampling definitivo)
    alto, ancho = img.shape[:2]
    hd_scale = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    
    # Suavizado de bordes profundos
    hd_blur = cv2.GaussianBlur(hd_scale, (5, 5), 0)
    _, final_hd = cv2.threshold(hd_blur, 200, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
