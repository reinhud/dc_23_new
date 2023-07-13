from src.inference.config.inference_config import InferenceConfig
from src.inference.timm_inferer import TimmInfererance

if __name__ == '__main__':
    infer_config = InferenceConfig()

    inferer = TimmInfererance(infer_config)

    #print(inferer.model.state_dict())
    # last row inferer pretrained false but no checkpint path given
    # -1.8739e-02, -1.2520e-02, -9.9689e-03, -2.2706e-02, -3.3813e-02],


    import torch

    checkpoint = torch.load("src/output/train/20230711-215544-resnet34-224/model_best.pth.tar")

    #print(checkpoint["state_dict"])
    # last row
    # -3.1077e-04, -4.4198e-02,  3.3924e-02, -1.5150e-02,  4.2471e-02],

    print(len(checkpoint["state_dict"]['fc.bias']))