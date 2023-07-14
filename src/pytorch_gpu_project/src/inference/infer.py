from PIL import Image
from src.inference.config.inference_config import InferenceConfig
from src.inference.timm_inferer_new import TimmInferer

if __name__ == '__main__':

    # Load image
    image1 = Image.open('src/data/raw/CN_dataset_04_23/data_types_example/1/CN_type_1_cn_coin_8022_p.jpg').convert('RGB')
    image2 = Image.open('src/data/raw/CN_dataset_04_23/data_types_example/3/CN_type_3_BNF_Platzhalter_cn_coin_11904_o.jpg').convert('RGB')

    images1 = [image1, image2]
    images2 = [image1]

    inferer = TimmInferer()

    # Load model
    model = inferer.load_model(model_name="resnet50")

    # Infer
    #prediction = inferer.infer(model, images1, topk=False)

    example_file_path = 'src/data/raw/CN_dataset_04_23/data_types_example/1/CN_type_1_cn_coin_8022_p.jpg'
    prediction = inferer.infer_from_file(file_path=example_file_path, model=model, images=images2, topk=False)

    print(prediction)

