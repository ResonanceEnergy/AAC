"""Full import test for all AAC project Python files."""
import os
import sys
import importlib

root = r'C:\dev\AAC_fresh'
sys.path.insert(0, root)

skip_dirs = {'__pycache__', 'node_modules', '.git', '.aac_venv', 'aac.egg-info', 'build', 'archive', 'old-tests'}
skip_files = {'_diagnostic.py', '_install_deps.py', '_full_import_test.py', 'setup_machine.py', 'conftest.py'}

ok = []
fail = []
total = 0

for dirpath, dirnames, fnames in os.walk(root):
    # Filter out skip directories
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    
    for fname in sorted(fnames):
        if not fname.endswith('.py') or fname in skip_files:
            continue
        
        filepath = os.path.join(dirpath, fname)
        relpath = os.path.relpath(filepath, root)
        total += 1
        
        # Convert file path to module path
        modpath = relpath.replace(os.sep, '.').replace('.py', '')
        
        try:
            importlib.import_module(modpath)
            ok.append(relpath)
        except Exception as e:
            ename = type(e).__name__
            emsg = str(e)[:120]
            fail.append((relpath, f"{ename}: {emsg}"))

print(f"\n{'='*60}")
print(f"IMPORT TEST RESULTS: {len(ok)}/{total} OK, {len(fail)} FAILED")
print(f"{'='*60}")

if fail:
    # Categorize failures
    categories = {}
    for path, err in fail:
        # Extract the error type
        if 'No module named' in err:
            missing = err.split("'")[1] if "'" in err else err
            cat = f"Missing: {missing}"
        elif 'cannot import name' in err:
            cat = "Bad import name"
        elif 'has no attribute' in err:
            cat = "Missing attribute"
        else:
            cat = err.split(':')[0]
        categories.setdefault(cat, []).append(path)
    
    print(f"\nFailures by category:")
    for cat, paths in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{len(paths)}] {cat}")
        for p in paths[:5]:
            print(f"       {p}")
        if len(paths) > 5:
            print(f"       ... and {len(paths)-5} more")

    print(f"\nAll failures:")
    for path, err in sorted(fail):
        print(f"  {path}")
        print(f"    {err}")
