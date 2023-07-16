from fastapi import APIRouter, File, UploadFile, HTTPException, status
from PIL import Image
from pytorch_service.src.inference.timm_inferer import TimmInferer
import io
from typing import Union

router = APIRouter()

timm_inferer = TimmInferer()


@router.post("/inference", name="Get predictions for an image")
async def infer(image: UploadFile = File(...), model_name: str = None, topk: Union[int, None] = None):
    image1 = Image.open('pytorch_service/src/data/raw/CN_dataset_04_23/data_types_example/1/CN_type_1_cn_coin_8022_p.jpg').convert('RGB')

    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")

    # Load model
    model = timm_inferer.load_model(model_name=model_name)

    # Infer
    prediction = timm_inferer.infer(model, [img], topk=topk)
    
    return {
        "Prediction": prediction
    }

