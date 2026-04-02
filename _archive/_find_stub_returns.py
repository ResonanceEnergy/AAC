"""Find methods whose entire body is just a single stub return."""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SKIP = {
    'archive', 'build', '.venv', '__pycache__', '.egg-info', 'aac.egg-info',
    'tests', '.git', 'docs', 'data', 'logs', 'reports', 'demos',
    'deployment', 'reddit', 'version_control', 'node_modules', '.vscode',
}

STUB_RETURNS = {'return {}', 'return []', 'return None', 'return 0', 'return 0.0', 'return False', 'return ""', "return ''"}

count = 0
results = []

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in sorted(files):
        if not f.endswith('.py'):
            continue
        fp = os.path.join(root, f)
        rel = os.path.relpath(fp, '.').replace('\\', '/')
        try:
            lines = open(fp, 'r', encoding='utf-8', errors='replace').readlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            s = line.strip()
            if s not in STUB_RETURNS:
                continue

            # Look backwards for def line
            def_line_idx = None
            for j in range(i - 1, max(i - 6, -1), -1):
                prev = lines[j].strip()
                if prev.startswith('def ') or prev.startswith('async def '):
                    def_line_idx = j
                    break
                # If we hit code that's not comment/docstring/blank, stop
                if prev and not prev.startswith('#') and not prev.startswith('"""') and not prev.startswith("'''") and prev != '"""' and prev != "'''":
                    break

            if def_line_idx is None:
                continue

            m = re.search(r'(?:async )?def\s+(\w+)', lines[def_line_idx])
            if not m:
                continue
            fname = m.group(1)

            # Skip dunders
            if fname.startswith('__') and fname.endswith('__'):
                continue

            # Check if the body between def and this return is ONLY docstrings/comments/blanks + the return
            has_real_code = False
            in_docstring = False
            for k in range(def_line_idx + 1, i):
                bl = lines[k].strip()
                if bl.startswith('"""') or bl.startswith("'''"):
                    if bl.count('"""') == 2 or bl.count("'''") == 2:
                        continue  # single-line docstring
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if not bl or bl.startswith('#'):
                    continue
                has_real_code = True
                break

            if not has_real_code:
                count += 1
                results.append(f'{rel}:{def_line_idx + 1}  {fname}  ->  {s}')

for r in sorted(results):
    print(r)

print(f'\nTotal stub-return methods: {count}')
