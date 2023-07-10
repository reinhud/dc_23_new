#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import logging
import os
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pandas as pd
import torch
from src.inference.inference_config.config_parser import _inference_parser
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

try:
    from apex import amp  # noqa: F401  # type: ignore

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError:
    has_functorch = False

has_compile = hasattr(torch, "compile")


_FMT_EXT = {
    "json": ".json",
    "json-record": ".json",
    "json-split": ".json",
    "parquet": ".parquet",
    "csv": ".csv",
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("inference")

parser = _inference_parser()


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert (
            has_native_amp
        ), "Please update PyTorch to a version with native AMP (or use APEX)."
        assert args.amp_dtype in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info("Running inference in mixed precision with native PyTorch AMP.")
    else:
        _logger.info("Running inference in float32. AMP not enabled.")

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        num_class_assert_error_msg = (
            "Model must have `num_classes` attr if not set on cmd line/config."
        )
        assert hasattr(model, "num_classes"), num_class_assert_error_msg
        args.num_classes = model.num_classes

    _logger.info(
        f"Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}"
    )

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        torchcompil_assert_error_msg = (
            ("A version of torch w/ torch.compile() is required for"),
        )
        ("--compile, possibly a nightly.")
        assert has_compile, torchcompil_assert_error_msg
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config["crop_pct"] = 1.0

    workers = 1 if "tfds" in args.dataset or "wds" in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        **data_config,
    )

    to_label = None
    if args.label_type in ("name", "description", "detail"):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == "name":
                to_label = lambda x: dataset_info.index_to_label_name(x)  # noqa: E731
            elif args.label_type == "detail":
                to_label = lambda x: dataset_info.index_to_description(  # noqa: E731
                    x, detailed=True
                )
            else:
                to_label = lambda x: dataset_info.index_to_description(x)  # noqa: E731
            to_label = np.vectorize(to_label)
        else:
            _logger.error(
                "Cannot deduce ImageNet subset from model, no labelling will be performed."
            )

    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    use_probs = args.output_type == "prob"
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            with amp_autocast():
                output = model(input)

            if use_probs:
                output = output.softmax(-1)

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    "Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        batch_idx, len(loader), batch_time=batch_time
                    )
                )

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    output_col = args.output_col or ("prob" if use_probs else "logit")
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f"{args.index_col}_{i}"] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f"{args.label_col}_{i}"] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f"{output_col}_{i}"] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f"{args.model}-{img_size}"

    if args.results_dir:
        results_filename = os.path.join(args.results_dir, results_filename)

    for fmt in args.results_format:
        save_results(df, results_filename, fmt)

    print("--result")
    print(df.set_index(args.filename_col).to_json(orient="index", indent=4))


def save_results(df, results_filename, results_format="csv", filename_col="filename"):
    results_filename += _FMT_EXT[results_format]
    if results_format == "parquet":
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == "json":
        df.set_index(filename_col).to_json(results_filename, indent=4, orient="index")
    elif results_format == "json-records":
        df.to_json(results_filename, lines=True, orient="records")
    elif results_format == "json-split":
        df.to_json(results_filename, indent=4, orient="split", index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    main()
