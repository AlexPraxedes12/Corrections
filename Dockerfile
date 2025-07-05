FROM python:3.10-slim

# Crear carpeta de trabajo
WORKDIR /code

# Copiar todo el contenido del proyecto
COPY . /code

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la API
EXPOSE 8000

# Ejecutar la API con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
