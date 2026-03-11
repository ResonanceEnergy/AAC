"""
Deploy BARREN WUFFET SKILL.md Files
=====================================

Generates and writes all 35 SKILL.md files to the OpenClaw skills directory.

Usage:
    python -m scripts.deploy_skills              # Deploy to default ~/.openclaw/workspace/skills/
    python -m scripts.deploy_skills --dir ./skills  # Deploy to custom directory
    python -m scripts.deploy_skills --dry-run       # Preview without writing
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openclaw_barren_wuffet_skills import (
    BARREN_WUFFET_SKILLS,
    generate_skill_md,
    write_all_skills,
)


def get_default_skills_dir() -> Path:
    """Get the default OpenClaw skills directory."""
    custom = os.getenv("OPENCLAW_SKILLS_DIR", "")
    if custom:
        return Path(custom)
    return Path.home() / ".openclaw" / "workspace" / "skills"


def deploy(target_dir: Path, dry_run: bool = False) -> int:
    """Deploy all SKILL.md files.
    
    Returns:
        Number of skills deployed.
    """
    print(f"\n{'=' * 60}")
    print(f"  BARREN WUFFET — SKILL.md Deployment")
    print(f"  Target: {target_dir}")
    print(f"  Skills: {len(BARREN_WUFFET_SKILLS)}")
    print(f"  Mode:   {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'=' * 60}\n")

    deployed = 0
    for slug, defn in BARREN_WUFFET_SKILLS.items():
        skill_dir = target_dir / slug
        skill_file = skill_dir / "SKILL.md"
        md = generate_skill_md(defn)

        if dry_run:
            print(f"  [DRY] {slug}/SKILL.md ({len(md)} bytes)")
        else:
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(md, encoding="utf-8")
            print(f"  [OK]  {slug}/SKILL.md ({len(md)} bytes)")

        deployed += 1

    print(f"\n{'=' * 60}")
    print(f"  Deployed: {deployed}/{len(BARREN_WUFFET_SKILLS)} skills")
    if dry_run:
        print(f"  (Dry run — no files written)")
    else:
        print(f"  Location: {target_dir}")
    print(f"{'=' * 60}\n")

    return deployed


def main():
    parser = argparse.ArgumentParser(
        description="Deploy BARREN WUFFET SKILL.md files to OpenClaw workspace"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="",
        help="Target directory (default: ~/.openclaw/workspace/skills/)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without writing files",
    )
    args = parser.parse_args()

    target = Path(args.dir) if args.dir else get_default_skills_dir()
    deploy(target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
