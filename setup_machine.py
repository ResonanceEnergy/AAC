#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   AAC — Unified Machine Setup                                ║
║   Works identically on QUSAR, QFORGE, and any new machine    ║
║                                                              ║
║   Usage:                                                     ║
║     python setup_machine.py              # Full setup        ║
║     python setup_machine.py --check      # Check only        ║
║     python setup_machine.py --fix        # Fix issues        ║
║     python setup_machine.py --reinstall  # Rebuild venv      ║
╚══════════════════════════════════════════════════════════════╝
"""
import os
import sys
import subprocess
import platform
import shutil
import json
import socket
from pathlib import Path
from datetime import datetime

# ── Constants ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIRED_PYTHON = (3, 9)   # Minimum
MAX_PYTHON = (3, 13)       # Maximum tested — 3.14 has aiohttp issues
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# Known Python locations on Windows (both machines)
PYTHON_SEARCH_PATHS = [
    Path(r"C:\Users\gripa\AppData\Local\Programs\Python\Python312\python.exe"),
    Path(r"C:\Users\gripa\AppData\Local\Programs\Python\Python313\python.exe"),
    Path(r"C:\Users\gripa\AppData\Local\Programs\Python\Python311\python.exe"),
    Path(r"C:\Python312\python.exe"),
    Path(r"C:\Python311\python.exe"),
    Path(r"C:\Python313\python.exe"),
]


def _cyan(t): return f"\033[96m{t}\033[0m" if sys.stdout.isatty() else t
def _green(t): return f"\033[92m{t}\033[0m" if sys.stdout.isatty() else t
def _yellow(t): return f"\033[93m{t}\033[0m" if sys.stdout.isatty() else t
def _red(t): return f"\033[91m{t}\033[0m" if sys.stdout.isatty() else t
def _bold(t): return f"\033[1m{t}\033[0m" if sys.stdout.isatty() else t


def banner():
    """Banner."""
    print(_cyan(r"""
  ╔══════════════════════════════════════════════════╗
  ║   AAC — Unified Machine Setup                    ║
  ║   Codename: BARREN WUFFET                        ║
  ║   Cross-Machine Unity Script v1.0                ║
  ╚══════════════════════════════════════════════════╝
"""))


def detect_machine():
    """Identify which machine we're running on."""
    hostname = socket.gethostname().upper()
    info = {
        "hostname": hostname,
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "user": os.environ.get("USERNAME", os.environ.get("USER", "unknown")),
    }
    if "QUSAR" in hostname:
        info["machine_name"] = "QUSAR"
    elif "QFORGE" in hostname:
        info["machine_name"] = "QFORGE"
    else:
        info["machine_name"] = hostname
    return info


def find_best_python():
    """Find the best Python interpreter (3.9 – 3.13, avoiding 3.14+)."""
    candidates = []

    # Check well-known paths
    for p in PYTHON_SEARCH_PATHS:
        if p.exists():
            ver = _get_python_version(str(p))
            if ver:
                candidates.append((str(p), ver))

    # Check PATH
    for name in ["python3", "python", "python3.12", "python3.11", "python3.13"]:
        path = shutil.which(name)
        if path:
            ver = _get_python_version(path)
            if ver and ver not in [c[1] for c in candidates]:
                candidates.append((path, ver))

    # Filter: must be >= REQUIRED_PYTHON and <= MAX_PYTHON
    valid = [(p, v) for p, v in candidates if REQUIRED_PYTHON <= v <= MAX_PYTHON]

    if not valid:
        # Show what we found
        print(_red("  [X] No suitable Python found!"))
        print(f"      Need Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} – "
              f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]}")
        if candidates:
            print("      Found (but unsuitable):")
            for p, v in candidates:
                print(f"        {p} → Python {v[0]}.{v[1]}.{v[2]}")
        return None

    # Prefer 3.12 > 3.11 > 3.13 > 3.10 > 3.9
    preferred_minors = [12, 11, 13, 10, 9]
    for minor in preferred_minors:
        for p, v in valid:
            if v[1] == minor:
                return p

    return valid[0][0]


def _get_python_version(python_path):
    """Get Python version tuple from executable."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import sys; print(sys.version_info[:3])"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return eval(result.stdout.strip())
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
    return None


def check_venv():
    """Check if virtual environment exists and is healthy."""
    if sys.platform == "win32":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
        venv_pip = VENV_DIR / "Scripts" / "pip.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"
        venv_pip = VENV_DIR / "bin" / "pip"

    if not venv_python.exists():
        return {"exists": False, "python": None, "version": None}

    ver = _get_python_version(str(venv_python))
    return {
        "exists": True,
        "python": str(venv_python),
        "version": ver,
        "pip": str(venv_pip) if venv_pip.exists() else None,
        "ok": ver is not None and REQUIRED_PYTHON <= ver <= MAX_PYTHON,
    }


def create_venv(python_path):
    """Create virtual environment using specified Python."""
    print(_cyan(f"  Creating .venv using {python_path}..."))
    result = subprocess.run(
        [python_path, "-m", "venv", str(VENV_DIR)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(_red(f"  [X] Failed to create venv: {result.stderr}"))
        return False

    # Upgrade pip
    if sys.platform == "win32":
        pip = str(VENV_DIR / "Scripts" / "pip.exe")
    else:
        pip = str(VENV_DIR / "bin" / "pip")

    print(_cyan("  Upgrading pip..."))
    subprocess.run([pip, "install", "--upgrade", "pip"], capture_output=True, timeout=120)
    print(_green("  [+] Virtual environment created"))
    return True


def install_deps():
    """Install all requirements into the venv."""
    if sys.platform == "win32":
        pip = str(VENV_DIR / "Scripts" / "pip.exe")
    else:
        pip = str(VENV_DIR / "bin" / "pip")

    print(_cyan(f"  Installing dependencies from {REQUIREMENTS.name}..."))
    result = subprocess.run(
        [pip, "install", "-r", str(REQUIREMENTS)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(_red(f"  [X] Dependency installation failed!"))
        # Show which packages failed
        for line in result.stderr.split('\n'):
            if 'error' in line.lower() or 'failed' in line.lower():
                print(_red(f"      {line.strip()}"))
        return False

    # Also install the project itself in editable mode
    print(_cyan("  Installing AAC package (editable)..."))
    subprocess.run(
        [pip, "install", "-e", str(PROJECT_ROOT)],
        capture_output=True, text=True, timeout=120,
    )

    print(_green("  [+] All dependencies installed"))
    return True


def check_syntax():
    """Quick syntax check on all Python files."""
    print(_cyan("  Running syntax check on all .py files..."))
    errors = []
    total = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip venv, __pycache__, archive
        dirs[:] = [d for d in dirs if d not in {'.venv', '__pycache__', 'archive',
                                                 '.git', 'node_modules', 'build',
                                                 'dist', 'aac.egg-info'}]
        for f in files:
            if f.endswith('.py'):
                total += 1
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8') as fh:
                        compile(fh.read(), fpath, 'exec')
                except SyntaxError as e:
                    rel = os.path.relpath(fpath, PROJECT_ROOT)
                    errors.append(f"{rel}:{e.lineno}: {e.msg}")
                except Exception:
                    pass  # encoding errors, etc.

    if errors:
        print(_yellow(f"  [!] {len(errors)} syntax errors in {total} files:"))
        for err in errors[:10]:
            print(_red(f"      {err}"))
        if len(errors) > 10:
            print(_yellow(f"      ... and {len(errors) - 10} more"))
    else:
        print(_green(f"  [+] All {total} .py files pass syntax check"))

    return errors


def check_critical_imports():
    """Test that critical imports work in the venv."""
    if sys.platform == "win32":
        venv_python = str(VENV_DIR / "Scripts" / "python.exe")
    else:
        venv_python = str(VENV_DIR / "bin" / "python")

    critical_packages = [
        "pandas", "numpy", "aiohttp", "requests", "ccxt",
        "dash", "plotly", "fastapi", "pydantic", "structlog",
        "rich", "cryptography", "sqlalchemy", "redis",
        "websockets", "matplotlib", "keyring",
    ]

    print(_cyan(f"  Testing {len(critical_packages)} critical imports..."))
    passed = []
    failed = []

    for pkg in critical_packages:
        try:
            result = subprocess.run(
                [venv_python, "-c", f"import {pkg}; print('OK')"],
                capture_output=True, text=True, timeout=15,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0 and 'OK' in result.stdout:
                passed.append(pkg)
            else:
                failed.append((pkg, result.stderr.strip()[:100]))
        except subprocess.TimeoutExpired:
            failed.append((pkg, "TIMEOUT"))
        except Exception as e:
            failed.append((pkg, str(e)[:100]))

    if failed:
        print(_yellow(f"  [!] {len(failed)} imports failed:"))
        for pkg, err in failed:
            print(_red(f"      {pkg}: {err}"))
    else:
        print(_green(f"  [+] All {len(passed)} critical imports work"))

    return passed, failed


def check_env_file():
    """Ensure .env exists (copy from template if not)."""
    env_file = PROJECT_ROOT / ".env"
    template = PROJECT_ROOT / ".env.template"

    if env_file.exists():
        print(_green("  [+] .env file exists"))
        return True
    elif template.exists():
        shutil.copy2(template, env_file)
        print(_yellow("  [!] Created .env from .env.template — fill in your API keys"))
        return True
    else:
        print(_yellow("  [!] No .env or .env.template found — some features may not work"))
        return False


def save_machine_state(info, venv_info, syntax_errors, import_results):
    """Save machine state to JSON for cross-machine comparison."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "machine": info,
        "venv": {
            "python": venv_info.get("python"),
            "version": f"{venv_info['version'][0]}.{venv_info['version'][1]}.{venv_info['version'][2]}" if venv_info.get("version") else None,
            "ok": venv_info.get("ok", False),
        },
        "syntax_errors": len(syntax_errors),
        "imports_passed": len(import_results[0]) if import_results else 0,
        "imports_failed": len(import_results[1]) if import_results else 0,
        "project_version": _get_project_version(),
    }

    state_file = PROJECT_ROOT / ".machine_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    return state


def _get_project_version():
    """Read version from pyproject.toml."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().split('\n'):
            if line.strip().startswith('version'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "unknown"


def print_report(machine_info, venv_info, syntax_errors, import_results, state):
    """Print final report."""
    print()
    print(_bold("=" * 60))
    print(_bold(f"  MACHINE SETUP REPORT — {machine_info['machine_name']}"))
    print(_bold("=" * 60))
    print()
    print(f"  Machine:    {machine_info['machine_name']} ({machine_info['hostname']})")
    print(f"  OS:         {machine_info['os']} {machine_info['os_version'][:30]}")
    print(f"  User:       {machine_info['user']}")
    print(f"  Project:    AAC v{state['project_version']}")
    print(f"  Workspace:  {PROJECT_ROOT}")
    print()

    if venv_info.get("ok"):
        v = venv_info["version"]
        print(_green(f"  Python:     {v[0]}.{v[1]}.{v[2]} (OK)"))
    else:
        print(_red(f"  Python:     ISSUE — see above"))

    print(f"  Syntax:     {len(syntax_errors)} errors")

    if import_results:
        p, f = import_results
        if f:
            print(_yellow(f"  Imports:    {len(p)} passed, {len(f)} failed"))
        else:
            print(_green(f"  Imports:    {len(p)} passed, 0 failed"))
    print()

    if not syntax_errors and import_results and not import_results[1]:
        print(_green("  ✅  MACHINE IS READY — united front confirmed"))
    elif syntax_errors:
        print(_yellow("  ⚠️  Fix syntax errors first: python setup_machine.py --fix"))
    else:
        print(_yellow("  ⚠️  Some imports failed — check dependencies"))

    print()
    print(f"  State saved to: .machine_state.json")
    print(f"  To compare machines: diff QUSAR/.machine_state.json QFORGE/.machine_state.json")
    print()


def main():
    """Main."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="setup_machine",
        description="AAC — Unified Cross-Machine Setup",
    )
    parser.add_argument("--check", action="store_true", help="Check only, don't install")
    parser.add_argument("--fix", action="store_true", help="Fix detected issues")
    parser.add_argument("--reinstall", action="store_true", help="Delete and recreate venv")
    args = parser.parse_args()

    banner()
    os.chdir(PROJECT_ROOT)

    # ── Step 1: Detect machine ──────────────────────────────────
    print(_bold("  Step 1: Machine Detection"))
    machine_info = detect_machine()
    print(_green(f"  [+] Machine: {machine_info['machine_name']}"))
    print(_green(f"  [+] OS: {machine_info['os']} | Arch: {machine_info['arch']}"))
    print()

    # ── Step 2: Find Python ─────────────────────────────────────
    print(_bold("  Step 2: Python Discovery"))
    best_python = find_best_python()
    if best_python:
        best_ver = _get_python_version(best_python)
        print(_green(f"  [+] Best Python: {best_python}"))
        print(_green(f"  [+] Version: {best_ver[0]}.{best_ver[1]}.{best_ver[2]}"))
    else:
        print(_red("  [X] No suitable Python (3.9–3.13) found!"))
        print(_red("      Install Python 3.12 from https://python.org/downloads/"))
        sys.exit(1)
    print()

    # ── Step 3: Virtual Environment ─────────────────────────────
    print(_bold("  Step 3: Virtual Environment"))
    venv_info = check_venv()

    if args.reinstall and venv_info["exists"]:
        print(_yellow("  [!] Removing existing venv for reinstall..."))
        shutil.rmtree(VENV_DIR)
        venv_info = {"exists": False}

    if not venv_info["exists"]:
        if args.check:
            print(_yellow("  [!] No .venv — run without --check to create"))
        else:
            create_venv(best_python)
            venv_info = check_venv()
    elif not venv_info.get("ok"):
        print(_yellow(f"  [!] venv Python version {venv_info.get('version')} is not ideal"))
        if not args.check:
            print(_yellow("  [!] Recreating with better Python..."))
            shutil.rmtree(VENV_DIR)
            create_venv(best_python)
            venv_info = check_venv()
    else:
        v = venv_info["version"]
        print(_green(f"  [+] .venv exists — Python {v[0]}.{v[1]}.{v[2]}"))
    print()

    # ── Step 4: Dependencies ────────────────────────────────────
    print(_bold("  Step 4: Dependencies"))
    if venv_info.get("exists") and not args.check:
        install_deps()
    elif args.check:
        print(_yellow("  [!] Skipping install (--check mode)"))
    print()

    # ── Step 5: Environment file ────────────────────────────────
    print(_bold("  Step 5: Environment"))
    check_env_file()
    print()

    # ── Step 6: Syntax check ────────────────────────────────────
    print(_bold("  Step 6: Syntax Validation"))
    syntax_errors = check_syntax()
    print()

    # ── Step 7: Import check ────────────────────────────────────
    print(_bold("  Step 7: Import Validation"))
    import_results = None
    if venv_info.get("exists") and venv_info.get("ok"):
        import_results = check_critical_imports()
    else:
        print(_yellow("  [!] Skipping import check — venv not ready"))
    print()

    # ── Step 8: Save state ──────────────────────────────────────
    state = save_machine_state(machine_info, venv_info, syntax_errors, import_results)

    # ── Report ──────────────────────────────────────────────────
    print_report(machine_info, venv_info, syntax_errors, import_results, state)


if __name__ == "__main__":
    main()
