"""Gap scanner v2 - finds real implementation gaps across the codebase."""
import os
import re
import sys
import io

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SKIP_DIRS = {
    'archive', 'build', '.venv', '__pycache__', '.egg-info', 'aac.egg-info',
    'tests', '.git', 'docs', 'data', 'logs', 'reports', 'demos',
    'deployment', 'reddit', 'version_control', 'node_modules',
}

TARGET_DIRS = [
    'shared', 'integrations', 'TradingExecution', 'monitoring', 'strategies',
    'services', 'agents', 'modules', 'models', 'core', 'CentralAccounting',
    'BigBrainIntelligence', 'CryptoIntelligence', 'tools', 'aac', 'src',
    'trading', 'SharedInfrastructure', 'agent_jonny_bravo_division', 'config',
]

gaps = []


def scan_file(filepath, rel):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            lines = content.splitlines()
    except Exception:
        return

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # TODO / FIXME markers
        if re.search(r'#\s*TODO\b', stripped, re.IGNORECASE):
            if 'survey' not in rel.lower() and 'gap' not in rel.lower() and '_scan' not in rel.lower():
                gaps.append((rel, '-', i, 'TODO', stripped[:120]))
        if re.search(r'#\s*FIXME\b', stripped, re.IGNORECASE):
            gaps.append((rel, '-', i, 'FIXME', stripped[:120]))

        # "coming soon" / "not yet implemented"
        if re.search(r'coming\s+soon|not\s+yet\s+implement', stripped, re.IGNORECASE):
            gaps.append((rel, '-', i, 'COMING_SOON', stripped[:120]))

        # raise NotImplementedError (not in abstract base)
        if 'raise NotImplementedError' in stripped:
            gaps.append((rel, '-', i, 'NOT_IMPL', stripped[:120]))

        # "pass" as sole body of a non-dunder, non-abstract, non-exception method
        # We detect this by checking: line is "pass", previous non-blank/docstring
        # line is a def, and it's not __init__/__repr__/etc.
        if stripped == 'pass':
            # Look backwards for the def line
            for j in range(i - 2, max(i - 6, -1), -1):
                if j < 0:
                    break
                prev = lines[j].strip()
                if prev.startswith('def ') or prev.startswith('async def '):
                    m = re.search(r'(?:async )?def\s+(\w+)', prev)
                    if m:
                        fname = m.group(1)
                        # Skip dunders, exception classes, abstract
                        if fname.startswith('__') and fname.endswith('__'):
                            break
                        # Check if it's an exception class body
                        if j > 0:
                            class_line = ''
                            for k in range(j - 1, max(j - 5, -1), -1):
                                cl = lines[k].strip()
                                if cl.startswith('class '):
                                    class_line = cl
                                    break
                            if 'Exception' in class_line or 'Error' in class_line:
                                break
                        gaps.append((rel, fname, j + 1, 'STUB_PASS', f'def {fname}: body is just pass'))
                    break
                elif prev and not prev.startswith('#') and not prev.startswith('"""') and not prev.startswith("'''"):
                    if not prev.startswith('"') and not prev.startswith("'"):
                        break

        # random.uniform/random/randint in signal/price generation (mock data)
        if re.search(r'random\.(uniform|random|randint|choice)\s*\(', stripped):
            if 'test' not in rel.lower() and 'conftest' not in rel.lower():
                ctx = stripped[:100]
                if any(w in ctx.lower() for w in ['signal', 'price', 'return', 'generate', 'score', 'confidence']):
                    gaps.append((rel, '-', i, 'MOCK_DATA', f'random instead of real: {ctx}'))


for d in TARGET_DIRS:
    dp = os.path.join('.', d)
    if not os.path.isdir(dp):
        continue
    for root, dirs, files in os.walk(dp):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in sorted(files):
            if f.endswith('.py'):
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, '.').replace('\\', '/')
                scan_file(fp, rel)

# Root level py files
for f in sorted(os.listdir('.')):
    if f.endswith('.py') and os.path.isfile(f) and not f.startswith('_'):
        scan_file(f, f)

# Print by category
cats = {}
for g in gaps:
    cats.setdefault(g[3], []).append(g)

for cat in sorted(cats.keys()):
    items = sorted(cats[cat], key=lambda x: x[0])
    print(f'\n=== {cat} ({len(items)}) ===')
    for g in items:
        print(f'  {g[0]}:{g[2]}  {g[1]}  {g[4]}')

print(f'\n{"=" * 60}')
print(f'TOTAL GAPS: {len(gaps)}')
for cat in sorted(cats.keys()):
    print(f'  {cat}: {len(cats[cat])}')
