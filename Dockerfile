# Imagen base con soporte para PyTorch + CUDA (usa esta si tienes GPU)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Establece el directorio de trabajo
WORKDIR /app

# Copia tus archivos
COPY . .

# Instala las dependencias (usa el requirements original de RETFound si está)
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install --no-cache-dir timm scikit-learn pandas matplotlib opencv-python fastapi uvicorn[standard] huggingface_hub

# Define variables de entorno si quieres
ENV CONFIG_PATH=configs/cdcl_effi_b3_p.json
ENV CHECKPOINT_PATH=models/RETFound_cfp_weights.pth

# Puerto para FastAPI
EXPOSE 8000

# Comando por defecto (cambia app.main por tu archivo si usas FastAPI)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
