import sys, re
from pathlib import Path

file = Path(__file__).resolve().parent.parent / 'imports2.txt'

modules = set()
if file.exists():
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('from .') or line.startswith('from ..') or line.startswith('import .'):
                continue
            if line.startswith('from '):
                parts = line.split()
                if len(parts) >= 2:
                    mod = parts[1]
                else:
                    continue
            elif line.startswith('import '):
                parts = line.split()
                if len(parts) >= 2:
                    mod = parts[1]
                else:
                    continue
            else:
                continue
            mod = mod.split('.')[0]
            modules.add(mod)

# Exclude standard libs
import sys as _sys
builtin = set(_sys.builtin_module_names)
stdlib_extra = {'sys','os','re','json','datetime','math','typing','pathlib','collections','itertools','asyncio','functools','logging','argparse','csv','urllib','http','subprocess','heapq','threading','queue','enum','socket','ssl'}
modules = sorted(m for m in modules if m not in builtin and m not in stdlib_extra and not m.startswith('_'))
with open('external_deps.txt','w') as out:
    for m in modules:
        out.write(m + '\n')
