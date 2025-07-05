from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import io
from models_vit import RETFound_mae

# Cargar modelo
model = RETFound_mae(
    img_size=224,
    num_classes=27,
    drop_path_rate=0.2,
    global_pool=True
)

checkpoint_path = "models/checkpoint-best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# Inicializar FastAPI
app = FastAPI(title="RETFound Disease Detection API")

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().tolist()

        return JSONResponse(content={"probabilities": probs})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
