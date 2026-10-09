"""User-installed instruction skills for Theriad.

Skills are bounded prompt-time guidance. Importing or enabling one does not
train a model and never grants tool, browser, payment, or execution authority.
"""

from .context import SkillContext, SkillContextPiece, build_skill_context, render_skill_context
from .importers import ImportResult, ParsedSkill, SkillImportError, parse_uploaded_skills
from .store import (
    MAX_DESCRIPTION_CHARS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_NAME_CHARS,
    SkillConflict,
    SkillNotFound,
    SkillRecord,
    SkillStore,
    SkillValidationError,
    get_enabled_skills,
)

__all__ = [
    "ImportResult",
    "MAX_DESCRIPTION_CHARS",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_NAME_CHARS",
    "ParsedSkill",
    "SkillConflict",
    "SkillContext",
    "SkillContextPiece",
    "SkillImportError",
    "SkillNotFound",
    "SkillRecord",
    "SkillStore",
    "SkillValidationError",
    "build_skill_context",
    "get_enabled_skills",
    "parse_uploaded_skills",
    "render_skill_context",
]
