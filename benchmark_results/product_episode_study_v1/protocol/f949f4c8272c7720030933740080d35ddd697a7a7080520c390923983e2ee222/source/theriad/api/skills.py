"""FastAPI CRUD and safe multipart import routes for Theriad skills."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from theriad.skills import (
    SkillConflict,
    SkillImportError,
    SkillNotFound,
    SkillStore,
    SkillValidationError,
    parse_uploaded_skills,
)
from theriad.skills.importers import MAX_UPLOAD_BYTES, MAX_UPLOAD_FILES
from theriad.skills.store import normalize_priority

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    instructions: str
    project_id: str | None = None
    enabled: bool = False
    priority: int = 100


class SkillUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    instructions: str | None = None
    project_id: str | None = None
    enabled: bool | None = None
    priority: int | None = None


def _validation_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


async def _read_uploads_bounded(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read multipart uploads without allocating beyond the importer limit."""
    if len(files) > MAX_UPLOAD_FILES:
        raise SkillImportError(
            f"too many uploaded files; maximum is {MAX_UPLOAD_FILES}"
        )

    remaining = MAX_UPLOAD_BYTES
    uploaded: list[tuple[str, bytes]] = []
    for item in files:
        payload = bytearray()
        while True:
            chunk = await item.read(min(64 * 1024, remaining + 1))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
            if remaining < 0:
                raise SkillImportError("uploads exceed the 10 MB total limit")
        uploaded.append((item.filename or "upload", bytes(payload)))
    return uploaded


@router.get("")
def list_skills(
    project_id: str | None = None,
    enabled: bool | None = None,
):
    store = SkillStore()
    if project_id is not None:
        try:
            project_id = store.validate_project_id(project_id)
        except SkillValidationError as exc:
            raise _validation_error(exc) from exc
    skills = store.list(
        project_id=project_id,
        effective=project_id is not None,
        enabled=enabled,
    )
    return {"skills": [skill.to_dict() for skill in skills], "count": len(skills)}


@router.post("", status_code=201)
def create_skill(body: SkillCreateRequest):
    try:
        skill = SkillStore().create(
            name=body.name,
            description=body.description,
            instructions=body.instructions,
            project_id=body.project_id,
            enabled=body.enabled,
            priority=body.priority,
        )
    except SkillConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SkillValidationError as exc:
        raise _validation_error(exc) from exc
    return {"skill": skill.to_dict()}


@router.post("/import/files", status_code=201)
@router.post("/import", status_code=201, include_in_schema=False)
async def import_skills(
    files: Annotated[list[UploadFile], File()],
    project_id: Annotated[str | None, Form()] = None,
    enabled: Annotated[bool, Form()] = False,
    priority: Annotated[int, Form()] = 100,
):
    try:
        uploaded = await _read_uploads_bounded(files)
        parsed = parse_uploaded_skills(uploaded)
    except SkillImportError as exc:
        raise _validation_error(exc) from exc

    store = SkillStore()
    try:
        project_id = store.validate_project_id(project_id)
        priority = normalize_priority(priority)
    except SkillValidationError as exc:
        raise _validation_error(exc) from exc
    created = []
    duplicates = 0
    warnings = list(parsed.warnings)
    if enabled:
        warnings.append(
            "Imported skills remain disabled until they are reviewed and enabled separately."
        )
    for item in parsed.skills:
        try:
            created.append(
                store.create(
                    name=item.name,
                    description=item.description,
                    instructions=item.instructions,
                    project_id=project_id,
                    enabled=False,
                    priority=priority,
                    source_type=item.source_type,
                    source_name=item.source_name,
                )
            )
        except SkillConflict:
            duplicates += 1
        except SkillValidationError as exc:
            warnings.append(f"Skipped {item.name}: {exc}")

    return {
        "skills": [skill.to_dict() for skill in created],
        "skills_created": len(created),
        "duplicates_skipped": duplicates,
        "files_skipped": parsed.files_skipped,
        "warnings": warnings,
        "enabled": False,
        "review_required": True,
    }


@router.get("/{skill_id}")
def get_skill(skill_id: str):
    skill = SkillStore().get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"skill '{skill_id}' not found")
    return {"skill": skill.to_dict()}


@router.patch("/{skill_id}")
def update_skill(skill_id: str, body: SkillUpdateRequest):
    fields = body.model_fields_set
    kwargs = {
        key: getattr(body, key)
        for key in ("name", "description", "instructions", "enabled", "priority")
        if key in fields
    }
    if "project_id" in fields:
        kwargs["project_id"] = body.project_id
    try:
        skill = SkillStore().update(skill_id, **kwargs)
    except SkillNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SkillConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SkillValidationError as exc:
        raise _validation_error(exc) from exc
    return {"skill": skill.to_dict()}


@router.delete("/{skill_id}")
def delete_skill(skill_id: str):
    if not SkillStore().delete(skill_id):
        raise HTTPException(status_code=404, detail=f"skill '{skill_id}' not found")
    return {"deleted": skill_id}
