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

def ages_subdiv_to_origin(sdage):
    result = []
    for age in sdage:
        if age < 2:
            result.append(0)
        elif age < 5:
            result.append(1)
        else:
            result.append(2)
    return result