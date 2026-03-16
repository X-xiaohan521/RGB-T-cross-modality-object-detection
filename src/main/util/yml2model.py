import math
import re
import yaml

from src.main.modules import *
import torch

def load_yml(file="data.yaml", append_filename=False):
    """Load YAML file to Python object with robust error handling.

    Args:
        file (str | Path): Path to YAML file.
        append_filename (bool): Whether to add filename to returned dict.

    Returns:
        (dict): Loaded YAML content.
    """
    assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"
    s = ""
    # Read file content
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()

    # Try loading YAML with fallback for problematic characters
    try:
        data = yaml.load(s, Loader=yaml.SafeLoader) or {}
    except Exception as e:
        # Remove problematic characters and retry
        s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        try:
            data = yaml.load(s, Loader=yaml.SafeLoader) or {}
        except Exception:
            raise ValueError(
                f"YAML syntax error in '{file}': {e}\nVerify YAML with https://ray.run/tools/yaml-formatter"
            ) from None

    # Check for accidental user-error None strings (should be 'null' in YAML)
    if "None" in data.values():
        data = {k: None if v == "None" else v for k, v in data.items()}

    if append_filename:
        data["yaml_file"] = str(file)
    return data

def make_divisible(x: int, divisor):
    """Return the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(model_config_dict, input_channels, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        model_config_dict (dict): Model dictionary.
        input_channels (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (torch.nn.Sequential): PyTorch model.
        (list): Sorted list of layer indices whose outputs need to be saved.
    """

    # Args
    max_channels = float("inf")
    nc, act, scales, end2end = (model_config_dict.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = model_config_dict.get("reg_max", 16)
    depth, width, kpt_shape = (model_config_dict.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = model_config_dict.get("scale")

    if verbose:
        print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    input_channels = [input_channels]
    layers, save, c2 = [], [], input_channels[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Conv,
            C3k2,
            SPPF,
            C2PSA
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            C3k2,
            C2PSA
        }
    )
    for i, (f, n, m, args) in enumerate(model_config_dict["backbone"] + model_config_dict["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = input_channels[f], args[0]
            if c2 != nc:  # if c2 != nc (e.g., Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True

        elif m is Concat:
            c2 = sum(input_channels[x] for x in f)
        elif m is OBB:
            args.extend([reg_max, end2end, [input_channels[x] for x in f]])
            m.legacy = legacy
        else:
            c2 = input_channels[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            print(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            input_channels = []
        input_channels.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)