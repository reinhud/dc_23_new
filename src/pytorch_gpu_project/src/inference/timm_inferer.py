import logging
from contextlib import suppress
from functools import partial

import torch
from timm.data import (
    ImageNetInfo,
    create_dataset,
    create_loader,
    infer_imagenet_subset,
    resolve_data_config,
)
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, set_jit_fuser, setup_default_logging


class TimmInfererance:
    def __init__(self, infer_config):
        self.infer_config = infer_config

        self.device = torch.device(self.infer_config.device)
        self.has_compile = hasattr(torch, "compile")
        self._memory_efficient_fusion = None

        self.amp_autocast = self._setup_amp()
        self.has_functorch = self._setup_functorch()
        self._setup_cuda()
        self._setup_fuser()

        self.model = self._setup_model()
        self.data_config = resolve_data_config(vars(self.infer_config), model=self.model)

    def _setup_amp(self):
        has_native_amp = False
        try:
            if getattr(torch.cuda.amp, "autocast") is not None:
                has_native_amp = True
        except AttributeError:
            pass

        # resolve AMP arguments based on PyTorch / Apex availability
        amp_autocast = suppress
        if self.infer_config.amp:
            assert (
                has_native_amp
            ), "Please update PyTorch to a version with native AMP (or use APEX)."
            assert self.infer_config.amp_dtype in ("float16", "bfloat16")
            amp_dtype = torch.bfloat16 if self.infer_config.amp_dtype == "bfloat16" else torch.float16
            amp_autocast = partial(torch.autocast, device_type=self.device.type, dtype=amp_dtype)
            print("Running inference in mixed precision with native PyTorch AMP.")
        else:
            print("Running inference in float32. AMP not enabled.")

        return amp_autocast

    def _setup_functorch(self):
        try:
            from functorch.compile import memory_efficient_fusion
            self._memory_efficient_fusion = memory_efficient_fusion
            print("Functorch found")
            return True
        except ImportError:
            print("Functorch not found")
            return False

    def _setup_cuda(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _setup_fuser(self):
        if self.infer_config.fuser:
            set_jit_fuser(self.infer_config.fuser)

    def _setup_model(self):
        in_chans = 3
        if self.infer_config.in_chans is not None:
            in_chans = self.infer_config.in_chans
        elif self.infer_config.input_size is not None:
            in_chans = self.infer_config.input_size[0]

        model = create_model(
            self.infer_config.model,
            num_classes=self.infer_config.num_classes,
            in_chans=in_chans,
            pretrained=self.infer_config.pretrained,
            checkpoint_path="src/output/train/20230711-215544-resnet34-224/model_best.pth.tar",     # TODO: make this default to best trained model for that architechture
        )

        data_config = resolve_data_config(vars(self.infer_config), model=model)
        test_time_pool = False
        if self.infer_config.test_pool:
            model, test_time_pool = apply_test_time_pool(model, data_config)

        model = model.to(self.device)
        model.eval()
        if self.infer_config.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.infer_config.torchscript:
            model = torch.jit.script(model)
        elif self.infer_config.torchcompile:
            torchcompil_assert_error_msg = (
                ("A version of torch w/ torch.compile() is required for"),
            )
            ("--compile, possibly a nightly.")
            assert self.has_compile, torchcompil_assert_error_msg
            torch._dynamo.reset()
            model = torch.compile(model, backend=self.infer_config.torchcompile)
        elif self.infer_config.aot_autograd:
            assert self.has_functorch, "functorch is needed for --aot-autograd"
            model = self.memory_efficient_fusion(model)

        if self.infer_config.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(self.infer_config.num_gpu)))

        print(f"Model {self.infer_config.model} created, param count: {sum([m.numel() for m in model.parameters()])}")

        return model

    def infer(self, inputs: list):

        # TODO: use the transforms used when training the model to transform the inputs

        all_indices = []
        all_labels = []
        all_outputs = []

        use_probs = self.infer_config.output_type == "prob"
        top_k = min(self.infer_config.topk, self.infer_config.num_classes)


        with torch.no_grad():
            for input in inputs:
                with self.amp_autocast():
                    output = self.model(input)

                if use_probs:
                    output = output.softmax(-1)

                if top_k:
                    output, indices = output.topk(top_k)
                    np_indices = indices.cpu().numpy()
                    all_indices.append(np_indices)

                    # TODO: get labels

                all_outputs.append(output.cpu().numpy())

        return all_indices, all_labels, all_outputs
