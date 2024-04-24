import argparse


def create_args(dict, description="Deep_EioU"):
    parser = argparse.ArgumentParser(description=description)
    for key, val in dict.items():
        parser.add_argument(f'--{key}', type=type(val), default=val, help=f"{key} description")

    return parser.parse_args()