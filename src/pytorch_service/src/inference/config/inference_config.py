import confuse
import yaml  # type: ignore


class InferenceConfig:
    """Class to hold all training base_configuration parameters."""

    BASE_CONFIG_PATH = "src/inference/config/base_config.yaml"

    base_config = confuse.Configuration("Base Inference Config")
    base_config.set_file(BASE_CONFIG_PATH)

    # TODO: add type to input from config
    def __init__(
        self,
        pretrained=base_config["pretrained"].get(),
        device=base_config["device"].get(),
        amp=base_config["amp"].get(),
        amp_dtype=base_config["amp_dtype"].get(),
        fuser=base_config["fuser"].get(),
        in_chans=base_config["in_chans"].get(),
        input_size=base_config["input_size"].get(),
        model=base_config["model"].get(),
        num_classes=base_config["num_classes"].get(),
        model_kwargs=base_config["model_kwargs"].get(),
        test_pool=base_config["test_pool"].get(),
        channels_last=base_config["channels_last"].get(),
        torchscript=base_config["torchscript"].get(),
        torchcompile=base_config["torchcompile"].get(),
        aot_autograd=base_config["aot_autograd"].get(),
        num_gpu=base_config["num_gpu"].get(),
        topk=base_config["topk"].get(),
    ):
        self.pretrained = pretrained
        self.device = device    # TODO: resolve cuda logic here
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.fuser = fuser
        self.in_chans = in_chans
        self.input_size = input_size
        self.model = model
        self.num_classes = num_classes
        self.model_kwargs = model_kwargs
        self.test_pool = test_pool
        self.channels_last = channels_last
        self.torchscript = torchscript
        self.torchcompile = torchcompile
        self.aot_autograd = aot_autograd
        self.num_gpu = num_gpu
        self.topk = topk