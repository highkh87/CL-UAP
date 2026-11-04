import argparse


def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)


def float_or_none(value):
    if value.lower() == 'none':
        return None
    return float(value)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")