import argparse

from timm.utils import ParseKwargs

parser = argparse.ArgumentParser(description="PyTorch ImageNet Inference")
parser.add_argument(
    "data",
    nargs="?",
    metavar="DIR",
    const=None,
    help="path to dataset (*deprecated*, use --data-dir)",
)
parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
parser.add_argument(
    "--dataset",
    metavar="NAME",
    default="",
    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)',
)
parser.add_argument(
    "--split",
    metavar="NAME",
    default="validation",
    help="dataset split (default: validation)",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="resnet50",
    help="model architecture (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--in-chans",
    type=int,
    default=None,
    metavar="N",
    help="Image input channels (default: None => 3)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224),"
    "uses model default if empty",
)
parser.add_argument(
    "--use-train-size",
    action="store_true",
    default=False,
    help="force use of train input size, even when test size is specified in pretrained cfg",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--crop-mode",
    default=None,
    type=str,
    metavar="N",
    help="Input image crop mode(squash, border, center). Model default if None.",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--num-classes", type=int, default=None, help="Number classes in dataset"
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)
parser.add_argument(
    "--log-freq",
    default=10,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--device", default="cuda", type=str, help="Device (accelerator) to use."
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use Native AMP for mixed precision training",
)
parser.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
parser.add_argument(
    "--fuser",
    default="",
    type=str,
    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
)
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    default=False,
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="Enable compilation w/ specified backend (default: inductor).",
)
scripting_group.add_argument(
    "--aot-autograd",
    default=False,
    action="store_true",
    help="Enable AOT Autograd support.",
)

parser.add_argument(
    "--results-dir", type=str, default=None, help="folder for output results"
)
parser.add_argument(
    "--results-file",
    type=str,
    default=None,
    help="results filename (relative to results-dir)",
)
parser.add_argument(
    "--results-format",
    type=str,
    nargs="+",
    default=["csv"],
    help='results format (one of "csv", "json", "json-split", "parquet")',
)
parser.add_argument(
    "--results-separate-col",
    action="store_true",
    default=False,
    help="separate output columns per result index.",
)
parser.add_argument(
    "--topk", default=1, type=int, metavar="N", help="Top-k to output to CSV"
)
parser.add_argument(
    "--fullname",
    action="store_true",
    default=False,
    help="use full sample name in output (not just basename).",
)
parser.add_argument(
    "--filename-col",
    type=str,
    default="filename",
    help="name for filename / sample name column",
)
parser.add_argument(
    "--index-col", type=str, default="index", help="name for output indices column(s)"
)
parser.add_argument(
    "--label-col", type=str, default="label", help="name for output indices column(s)"
)
parser.add_argument(
    "--output-col", type=str, default=None, help="name for logit/probs output column(s)"
)
parser.add_argument(
    "--output-type",
    type=str,
    default="prob",
    help='output type colum ("prob" for probabilities, "logit" for raw logits)',
)
parser.add_argument(
    "--label-type",
    type=str,
    default="description",
    help='type of label to output, one of  "none", "name", "description", "detailed"',
)
parser.add_argument(
    "--include-index",
    action="store_true",
    default=False,
    help="include the class index in results",
)
parser.add_argument(
    "--exclude-output",
    action="store_true",
    default=False,
    help="exclude logits/probs from results, just indices. topk must be set !=0.",
)


def _inference_parser():
    return parser
