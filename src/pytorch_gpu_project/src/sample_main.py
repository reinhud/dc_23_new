import torch


def main():
    """Shows if Cuda was set up correctly."""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.backends.mkl.is_available():", torch.backends.mkl.is_available())
    print("torch.backends.cudnn.is_available():", torch.backends.cudnn.is_available())
    print("torch.backends.cuda.is_built():", torch.backends.cuda.is_built())
    print("torch.backends.mkldnn.is_available():", torch.backends.mkldnn.is_available())


if __name__ == "__main__":
    main()

# TODO: Add logging
# TODO: save best models and changes according to architecture, make it smaller, only load model architechture and params?  # noqa: E501
# TODO: write inference loop
# TODO: make test setup where couple different architecture setups are tested
# TODO: integrate a config for training runs
# TODO: add more data augmentation
# TODO: add more models
# TODO: builde api for inference
# TODO: docs schreiben
# TODO: optimize models like so? https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/
# TODO: add logic to optionally save all trianing runs or only when model performed better, or only save some params like summary and args?  # noqa: E501
# TODO: make logged output of training run prettier (tqdm bar for epochs and subbar for epoch?)
