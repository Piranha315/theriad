"""Account billing and entitlement API."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from theriad.api.auth import get_request_user
from theriad.api.auth_context import require_auth_context
from theriad.billing.service import (
    BillingUnavailable,
    apply_webhook_event,
    billing_status_for,
    create_checkout_session,
    create_portal_session,
    entitlements_for,
    get_billing_config,
    verify_webhook_signature,
)

router = APIRouter(prefix="/api/billing", tags=["billing"])
entitlements_router = APIRouter(prefix="/api", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan_id: str = Field(pattern="^(starter|pro)$")
    interval: str = Field(default="monthly", pattern="^(monthly|annual)$")


def _account(request: Request):
    context = require_auth_context(request)
    if context.principal_type == "guest":
        raise HTTPException(
            status_code=403,
            detail="Register an account before starting a paid subscription.",
        )
    user = get_request_user(request)
    if not user or not context.organization_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return context, user


@router.get("/status")
def billing_status(request: Request):
    context = require_auth_context(request)
    return {
        **billing_status_for(context.organization_id),
        "principal_type": context.principal_type,
    }


@router.get("/entitlements")
def billing_entitlements(request: Request):
    context = require_auth_context(request)
    return entitlements_for(
        organization_id=context.organization_id,
        user_id=context.user_id,
        guest=context.principal_type == "guest",
    )


@entitlements_router.get("/entitlements")
def entitlements_alias(request: Request):
    """Short public contract kept alongside the namespaced billing route."""
    return billing_entitlements(request)


@router.post("/checkout")
async def billing_checkout(body: CheckoutRequest, request: Request):
    context, user = _account(request)
    if not get_billing_config().live_ready:
        return JSONResponse(
            status_code=503,
            content={
                "error": "billing_not_configured",
                "billing_status": "policy_only",
                "message": "Stripe checkout is not configured for this deployment.",
            },
        )
    try:
        return await create_checkout_session(
            organization_id=context.organization_id,
            user_id=context.user_id,
            email=user.get("email"),
            plan_id=body.plan_id,
            interval=body.interval,
        )
    except BillingUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": "billing_unavailable", "message": str(exc)})


@router.post("/portal")
async def billing_portal(request: Request):
    context, _user = _account(request)
    try:
        return await create_portal_session(organization_id=context.organization_id)
    except BillingUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": "billing_unavailable", "message": str(exc)})


@router.post("/webhook")
async def billing_webhook(request: Request):
    payload = await request.body()
    if not verify_webhook_signature(payload, request.headers.get("stripe-signature")):
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature.")
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe event payload.") from exc
    try:
        processed, status = apply_webhook_event(event)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"received": True, "processed": processed, "status": status}
