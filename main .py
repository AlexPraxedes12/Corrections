from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import io
import logging
from models_vit import RETFound_mae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Received prediction request: %s (%d bytes)", file.filename, len(image_bytes))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        logger.info("Tensor shape after transforms: %s", tuple(input_tensor.shape))

        with torch.no_grad():
            try:
                outputs = model(input_tensor)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception("Model inference failed")
                return JSONResponse(status_code=500, content={"error": str(e)})
            probs = torch.sigmoid(outputs).squeeze().tolist()

        return JSONResponse(content={"probabilities": probs})

    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
