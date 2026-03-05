"""Tests for OpenClaw BARREN WUFFET Skills Module.

Validates the 35-skill definitions, RESEARCH_INTEL data, utility functions,
and SKILL.md generation for the BARREN WUFFET OpenClaw integration.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openclaw_barren_wuffet_skills import (
    BARREN_WUFFET_SKILLS,
    RESEARCH_INTEL,
    generate_skill_md,
    get_research_intel,
    get_safe_trading_checklist,
    get_scam_alerts,
    get_skill_count,
    get_skill_definition,
    get_skill_names,
    get_skills_by_category,
)


# ── Skill Definitions ─────────────────────────────────────────────────────


class TestSkillDefinitions:
    """Validate the BARREN WUFFET skill entries."""

    def test_skill_count(self):
        """Must have the expected number of skills."""
        assert get_skill_count() == len(BARREN_WUFFET_SKILLS)
        assert get_skill_count() >= 35

    def test_all_skills_have_required_fields(self):
        """Every skill must have name and description."""
        required_fields = {"name", "description"}
        for slug, defn in BARREN_WUFFET_SKILLS.items():
            missing = required_fields - set(defn.keys())
            assert not missing, f"Skill '{slug}' missing fields: {missing}"

    def test_slug_prefix(self):
        """All skill slugs must start with 'bw-'."""
        for slug in BARREN_WUFFET_SKILLS:
            assert slug.startswith("bw-"), f"Slug '{slug}' does not start with 'bw-'"

    def test_skill_names_list(self):
        """get_skill_names() returns all slugs."""
        names = get_skill_names()
        assert len(names) == get_skill_count()
        assert all(isinstance(n, str) for n in names)

    def test_get_skill_definition_found(self):
        """Can retrieve a known skill by slug."""
        defn = get_skill_definition("bw-market-intelligence")
        assert defn is not None
        assert "description" in defn

    def test_get_skill_definition_not_found(self):
        """Returns None for unknown slug."""
        assert get_skill_definition("bw-nonexistent") is None

    def test_categories_present(self):
        """All expected category lookups return results."""
        expected_categories = [
            "core", "trading", "crypto", "finance",
            "wealth", "advanced", "powerups",
        ]
        for cat in expected_categories:
            skills = get_skills_by_category(cat)
            assert len(skills) >= 1, f"Category '{cat}' returned no skills"

    def test_get_skills_by_category(self):
        """get_skills_by_category returns non-empty dict for known category."""
        core = get_skills_by_category("core")
        assert len(core) >= 5


# ── SKILL.md Generation ───────────────────────────────────────────────────


class TestSkillMdGeneration:
    """Validate SKILL.md output format."""

    def test_generate_skill_md_returns_string(self):
        defn = get_skill_definition("bw-trading-signals")
        md = generate_skill_md(defn)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_md_contains_yaml_frontmatter(self):
        defn = get_skill_definition("bw-risk-monitor")
        md = generate_skill_md(defn)
        assert md.startswith("---"), "SKILL.md must start with YAML frontmatter"
        assert md.count("---") >= 2

    def test_md_contains_skill_name(self):
        defn = get_skill_definition("bw-crypto-intel")
        md = generate_skill_md(defn)
        assert "bw-crypto-intel" in md.lower() or defn["name"].lower() in md.lower()


# ── RESEARCH_INTEL ─────────────────────────────────────────────────────────


class TestResearchIntel:
    """Validate the RESEARCH_INTEL dictionary."""

    def test_research_intel_domains(self):
        """Must have core intelligence domains."""
        expected_core = {"trading_modes", "investor_patterns", "crypto_patterns",
                    "scam_intelligence", "reliability"}
        assert expected_core.issubset(set(RESEARCH_INTEL.keys())), (
            f"Missing core domains: {expected_core - set(RESEARCH_INTEL.keys())}"
        )

    def test_get_research_intel_found(self):
        domain = get_research_intel("trading_modes")
        assert domain is not None
        assert isinstance(domain, dict)

    def test_get_research_intel_not_found(self):
        assert get_research_intel("nonexistent_domain") is None


# ── Scam Alerts ────────────────────────────────────────────────────────────


class TestScamAlerts:
    """Validate scam detection outputs."""

    def test_scam_alerts_returns_list(self):
        alerts = get_scam_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) > 0

    def test_scam_alerts_mention_known_scams(self):
        alerts_text = " ".join(get_scam_alerts()).lower()
        # Must reference at least one known scam token
        assert any(token in alerts_text for token in ["clawd", "openclaw", "fclaw"]), (
            "Scam alerts should mention at least one known scam token"
        )


# ── Safe Trading Checklist ─────────────────────────────────────────────────


class TestSafeTradingChecklist:
    """Validate safe trading checklist output."""

    def test_checklist_returns_list(self):
        checklist = get_safe_trading_checklist()
        assert isinstance(checklist, list)
        assert len(checklist) >= 3

    def test_checklist_items_are_strings(self):
        for item in get_safe_trading_checklist():
            assert isinstance(item, str)
            assert len(item) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
