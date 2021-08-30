from pathlib import Path
import json
from collections import OrderedDict

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def update_argument(args, configs):
    for arg in configs:
        if arg in args:
            setattr(args, arg, configs[arg])
        else:
            raise ValueError(f"no argument {arg}")
    return args