"""Safe text-only import for standalone and bundled Theriad skills."""

from __future__ import annotations

import io
import re
import stat
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import yaml

from .store import (
    MAX_INSTRUCTIONS_CHARS,
    normalize_description,
    normalize_instructions,
    normalize_name,
)

MAX_UPLOAD_FILES = 500
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_ZIP_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
MAX_TEXT_FILE_BYTES = 256 * 1024
MAX_COMPRESSION_RATIO = 100
MAX_FRONTMATTER_BYTES = 8 * 1024
MAX_REFERENCE_FILES_PER_SKILL = 20
MAX_WARNINGS = 100

TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
ARCHIVE_EXTENSIONS = {".zip"}
_FRONTMATTER_KEYS = {"name", "description", "version", "license"}


class SkillImportError(ValueError):
    """The upload cannot be interpreted safely as one or more skills."""


@dataclass(frozen=True)
class ParsedSkill:
    name: str
    description: str
    instructions: str
    source_type: str
    source_name: str


@dataclass(frozen=True)
class ImportResult:
    skills: list[ParsedSkill] = field(default_factory=list)
    files_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _VirtualFile:
    bundle_id: str
    path: PurePosixPath
    payload: bytes
    source_type: str
    source_name: str


def _warning(warnings: list[str], message: str) -> None:
    if len(warnings) < MAX_WARNINGS:
        warnings.append(message)


def _safe_path(filename: str) -> PurePosixPath:
    filename = unicodedata.normalize("NFC", str(filename or ""))
    if not filename or "\x00" in filename or "\\" in filename:
        raise SkillImportError("file path is empty or contains unsafe characters")
    if len(filename) > 1_024:
        raise SkillImportError("file path exceeds the 1,024 character limit")
    if any(unicodedata.category(char) in {"Cc", "Cf"} for char in filename):
        raise SkillImportError("file path contains control or formatting characters")
    path = PurePosixPath(filename)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SkillImportError(f"unsafe file path: {filename}")
    if any(len(part) > 255 for part in path.parts):
        raise SkillImportError("a file path component exceeds the 255 character limit")
    return path


def _decode_text(payload: bytes, path: PurePosixPath) -> str:
    if len(payload) > MAX_TEXT_FILE_BYTES:
        raise SkillImportError(
            f"{path}: text files must be at most {MAX_TEXT_FILE_BYTES // 1024} KB"
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SkillImportError(f"{path}: expected UTF-8 text") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text


def _expand_zip(
    filename: str,
    payload: bytes,
    bundle_id: str,
    warnings: list[str],
) -> tuple[list[_VirtualFile], int, int, int]:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise SkillImportError(f"{filename}: archive exceeds the 10 MB upload limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise SkillImportError(f"{filename}: invalid ZIP archive") from exc

    files: list[_VirtualFile] = []
    skipped = 0
    seen_paths: set[str] = set()
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise SkillImportError(
                f"{filename}: archive contains more than {MAX_ZIP_ENTRIES} entries"
            )
        total_uncompressed = sum(max(0, info.file_size) for info in infos)
        if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise SkillImportError(f"{filename}: archive expands beyond the 25 MB limit")

        for info in infos:
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise SkillImportError(f"{filename}: encrypted ZIP members are not supported")
            unix_mode = info.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                skipped += 1
                _warning(warnings, f"Skipped symlink {info.filename}")
                continue
            try:
                path = _safe_path(info.filename)
            except SkillImportError:
                skipped += 1
                _warning(warnings, f"Skipped unsafe archive path {info.filename!r}")
                continue
            normalized_key = path.as_posix().casefold()
            if normalized_key in seen_paths:
                raise SkillImportError(f"{filename}: duplicate normalized path {path}")
            seen_paths.add(normalized_key)

            if info.file_size and info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
                raise SkillImportError(
                    f"{filename}: suspicious compression ratio for {path}"
                )
            suffix = path.suffix.lower()
            if suffix in ARCHIVE_EXTENSIONS:
                skipped += 1
                _warning(warnings, f"Skipped nested archive {path}")
                continue
            if suffix not in TEXT_EXTENSIONS:
                skipped += 1
                _warning(warnings, f"Ignored non-instruction file {path}")
                continue
            if info.file_size > MAX_TEXT_FILE_BYTES:
                skipped += 1
                _warning(warnings, f"Skipped oversized text file {path}")
                continue
            try:
                member_payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise SkillImportError(f"{filename}: could not read ZIP member {path}") from exc
            files.append(
                _VirtualFile(
                    bundle_id=bundle_id,
                    path=path,
                    payload=member_payload,
                    source_type="zip",
                    source_name=filename,
                )
            )
    return files, skipped, len(infos), total_uncompressed


def _frontmatter(text: str, path: PurePosixPath) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    closing = text.find("\n---\n", 4)
    if closing < 0 or len(text[4:closing].encode("utf-8")) > MAX_FRONTMATTER_BYTES:
        raise SkillImportError(f"{path}: malformed or oversized YAML frontmatter")
    raw = text[4:closing]
    try:
        loaded: Any = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise SkillImportError(f"{path}: invalid YAML frontmatter") from exc
    if not isinstance(loaded, dict):
        raise SkillImportError(f"{path}: YAML frontmatter must be a mapping")
    metadata: dict[str, str] = {}
    for key in _FRONTMATTER_KEYS:
        if key not in loaded:
            continue
        value = loaded[key]
        if not isinstance(value, str | int | float | bool):
            raise SkillImportError(f"{path}: frontmatter field '{key}' must be a scalar")
        metadata[key] = str(value).strip()
    return metadata, text[closing + 5 :]


def _derive_name(metadata: dict[str, str], body: str, path: PurePosixPath) -> str:
    if metadata.get("name"):
        return normalize_name(metadata["name"])
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", body)
    if heading:
        return normalize_name(heading.group(1))
    if path.name.casefold() == "skill.md" and path.parent.name:
        return normalize_name(path.parent.name.replace("-", " ").replace("_", " "))
    return normalize_name(path.stem.replace("-", " ").replace("_", " "))


def _is_reference_for(candidate: _VirtualFile, skill_file: _VirtualFile) -> bool:
    if candidate.bundle_id != skill_file.bundle_id or candidate.path == skill_file.path:
        return False
    if candidate.path.name.casefold() == "skill.md":
        return False
    try:
        relative = candidate.path.relative_to(skill_file.path.parent)
    except ValueError:
        return False
    return len(relative.parts) >= 2 and relative.parts[0].casefold() == "references"


def _parse_skill_file(skill_file: _VirtualFile, all_files: list[_VirtualFile], warnings: list[str]) -> ParsedSkill:
    text = _decode_text(skill_file.payload, skill_file.path)
    metadata, body = _frontmatter(text, skill_file.path)
    body = body.strip()
    if not body:
        raise SkillImportError(f"{skill_file.path}: skill instructions are empty")

    references = sorted(
        (candidate for candidate in all_files if _is_reference_for(candidate, skill_file)),
        key=lambda item: item.path.as_posix().casefold(),
    )
    if len(references) > MAX_REFERENCE_FILES_PER_SKILL:
        for item in references[MAX_REFERENCE_FILES_PER_SKILL:]:
            _warning(warnings, f"Skipped extra reference {item.path}")
        references = references[:MAX_REFERENCE_FILES_PER_SKILL]

    instructions = body
    for reference in references:
        reference_text = _decode_text(reference.payload, reference.path).strip()
        if not reference_text:
            continue
        addition = f"\n\n## Imported reference: {reference.path.name}\n{reference_text}"
        if len(instructions) + len(addition) > MAX_INSTRUCTIONS_CHARS:
            _warning(warnings, f"Skipped reference {reference.path}: skill text limit reached")
            continue
        instructions += addition

    return ParsedSkill(
        name=_derive_name(metadata, body, skill_file.path),
        description=normalize_description(metadata.get("description", "")),
        instructions=normalize_instructions(instructions),
        source_type=skill_file.source_type,
        source_name=skill_file.source_name,
    )


def parse_uploaded_skills(files: list[tuple[str, bytes]]) -> ImportResult:
    """Parse uploads without extracting archives or retaining executable files."""
    if not files:
        raise SkillImportError("at least one skill file is required")
    if len(files) > MAX_UPLOAD_FILES:
        raise SkillImportError(f"too many uploaded files; maximum is {MAX_UPLOAD_FILES}")
    if sum(len(payload) for _, payload in files) > MAX_UPLOAD_BYTES:
        raise SkillImportError("uploads exceed the 10 MB total limit")

    virtual_files: list[_VirtualFile] = []
    warnings: list[str] = []
    skipped = 0
    seen_direct: set[str] = set()
    archive_entries = 0
    archive_uncompressed_bytes = 0

    for index, (filename, payload) in enumerate(files):
        try:
            path = _safe_path(filename)
        except SkillImportError:
            skipped += 1
            _warning(warnings, f"Skipped unsafe upload path {filename!r}")
            continue
        key = path.as_posix().casefold()
        if key in seen_direct:
            raise SkillImportError(f"duplicate normalized upload path {path}")
        seen_direct.add(key)
        suffix = path.suffix.lower()
        if suffix in ARCHIVE_EXTENSIONS:
            expanded, zip_skipped, zip_entries, zip_uncompressed = _expand_zip(
                filename, payload, f"zip:{index}", warnings
            )
            archive_entries += zip_entries
            archive_uncompressed_bytes += zip_uncompressed
            if archive_entries > MAX_ZIP_ENTRIES:
                raise SkillImportError(
                    f"uploaded archives contain more than {MAX_ZIP_ENTRIES} entries in total"
                )
            if archive_uncompressed_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise SkillImportError(
                    "uploaded archives expand beyond the 25 MB total limit"
                )
            virtual_files.extend(expanded)
            skipped += zip_skipped
            continue
        if suffix not in TEXT_EXTENSIONS:
            skipped += 1
            _warning(warnings, f"Ignored non-instruction file {path}")
            continue
        if len(payload) > MAX_TEXT_FILE_BYTES:
            skipped += 1
            _warning(warnings, f"Skipped oversized text file {path}")
            continue
        virtual_files.append(
            _VirtualFile(
                bundle_id="direct",
                path=path,
                payload=payload,
                source_type=(
                    "folder"
                    if len(path.parts) > 1
                    else "markdown"
                    if suffix in {".md", ".markdown"}
                    else "text"
                ),
                source_name=path.as_posix(),
            )
        )

    skill_files = [item for item in virtual_files if item.path.name.casefold() == "skill.md"]

    # A directly selected top-level Markdown/TXT file is a complete skill. For a
    # ZIP without SKILL.md, accept a single root instruction document; multi-file
    # bundles must identify skill roots explicitly to avoid ingesting READMEs.
    for item in virtual_files:
        if item in skill_files or len(item.path.parts) != 1:
            continue
        if item.source_type != "zip":
            skill_files.append(item)

    for bundle_id in {item.bundle_id for item in virtual_files if item.source_type == "zip"}:
        bundle = [item for item in virtual_files if item.bundle_id == bundle_id]
        if not any(item.path.name.casefold() == "skill.md" for item in bundle):
            roots = [item for item in bundle if len(item.path.parts) == 1]
            if len(roots) == 1:
                skill_files.append(roots[0])

    # De-duplicate candidates while preserving upload/archive order.
    unique_candidates: list[_VirtualFile] = []
    seen_candidates: set[tuple[str, str]] = set()
    for item in skill_files:
        key = (item.bundle_id, item.path.as_posix().casefold())
        if key not in seen_candidates:
            unique_candidates.append(item)
            seen_candidates.add(key)

    if not unique_candidates:
        raise SkillImportError(
            "no skills found; upload a Markdown/TXT file or a folder/ZIP containing SKILL.md"
        )

    parsed = [_parse_skill_file(item, virtual_files, warnings) for item in unique_candidates]
    return ImportResult(skills=parsed, files_skipped=skipped, warnings=warnings)
