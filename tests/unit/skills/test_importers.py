from __future__ import annotations

import io
import stat
import zipfile

import pytest

from theriad.skills.importers import SkillImportError, parse_uploaded_skills


def _zip(entries: dict[str, bytes | str], compression=zipfile.ZIP_STORED) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return payload.getvalue()


def test_imports_standalone_markdown_with_frontmatter():
    result = parse_uploaded_skills([
        (
            "review.md",
            b"---\nname: Careful Reviewer\ndescription: Review before answering\n---\n# Rules\nCheck assumptions.",
        )
    ])
    skill = result.skills[0]
    assert skill.name == "Careful Reviewer"
    assert skill.description == "Review before answering"
    assert "Check assumptions" in skill.instructions
    assert skill.source_type == "markdown"


def test_folder_uses_skill_md_and_bounded_references_but_ignores_scripts():
    result = parse_uploaded_skills([
        ("careful/SKILL.md", b"---\nname: Careful\n---\nFollow the checklist."),
        ("careful/references/checklist.md", b"1. Verify facts.\n2. Cite sources."),
        ("careful/scripts/run.py", b"print('must never execute')"),
    ])
    assert len(result.skills) == 1
    assert "Imported reference: checklist.md" in result.skills[0].instructions
    assert "Verify facts" in result.skills[0].instructions
    assert "must never execute" not in result.skills[0].instructions
    assert result.files_skipped == 1


def test_zip_imports_multiple_skill_roots_and_skips_traversal_and_binaries():
    payload = _zip({
        "one/SKILL.md": "# One\nDo one thing.",
        "two/SKILL.md": "# Two\nDo another thing.",
        "../escape/SKILL.md": "# Escape\nUnsafe.",
        "one/image.png": b"not an image",
        "one/nested.zip": b"not really a zip",
    })
    result = parse_uploaded_skills([("bundle.zip", payload)])
    assert {skill.name for skill in result.skills} == {"One", "Two"}
    assert result.files_skipped == 3
    assert any("unsafe archive path" in warning for warning in result.warnings)
    assert all("Unsafe" not in skill.instructions for skill in result.skills)


def test_zip_bomb_like_compression_ratio_is_rejected():
    payload = _zip(
        {"bomb/SKILL.md": "# Bomb\n" + ("x" * 200_000)},
        compression=zipfile.ZIP_DEFLATED,
    )
    with pytest.raises(SkillImportError, match="compression ratio"):
        parse_uploaded_skills([("bomb.zip", payload)])


def test_malformed_frontmatter_and_non_utf8_are_rejected():
    with pytest.raises(SkillImportError, match="frontmatter"):
        parse_uploaded_skills([("bad.md", b"---\nname: [broken\n---\nbody")])
    with pytest.raises(SkillImportError, match="UTF-8"):
        parse_uploaded_skills([("bad.txt", b"\xff\xfe")])


def test_multi_file_bundle_requires_skill_md():
    with pytest.raises(SkillImportError, match="no skills found"):
        parse_uploaded_skills([
            ("folder/README.md", b"Project readme"),
            ("folder/notes.txt", b"Notes"),
        ])


def test_duplicate_normalized_zip_paths_are_rejected():
    payload = _zip({"one/SKILL.md": "# One\nA", "ONE/skill.md": "# Other\nB"})
    with pytest.raises(SkillImportError, match="duplicate normalized path"):
        parse_uploaded_skills([("duplicate.zip", payload)])


def test_zip_symlink_is_skipped_and_never_read_as_reference():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("safe/SKILL.md", "# Safe\nFollow safe instructions.")
        link = zipfile.ZipInfo("safe/references/link.md")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../secret.md")

    result = parse_uploaded_skills([("links.zip", payload.getvalue())])
    assert len(result.skills) == 1
    assert "secret.md" not in result.skills[0].instructions
    assert result.files_skipped == 1
    assert any("symlink" in warning for warning in result.warnings)
