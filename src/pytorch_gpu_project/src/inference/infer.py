from PIL import Image
from src.inference.config.inference_config import InferenceConfig
from src.inference.timm_inferer_new import TimmInferer

if __name__ == '__main__':

    # Load image
    image1 = Image.open('src/data/raw/CN_dataset_04_23/data_types_example/1/CN_type_1_cn_coin_8022_p.jpg').convert('RGB')
    image2 = Image.open('src/data/raw/CN_dataset_04_23/data_types_example/3/CN_type_3_BNF_Platzhalter_cn_coin_11904_o.jpg').convert('RGB')

    images = [image1, image2]

    inferer = TimmInferer()

    # Load model
    model = inferer.load_model(model_name="resnet50")

    # Infer
    prediction = inferer.infer(model, images, topk=5)

    print(prediction)

