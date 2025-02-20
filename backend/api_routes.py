#!/usr/bin/env python3
"""Minimal API routes for the SutazAI backend."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
def get_status():
    return {"status": "running"}


# Additional API routes can be defined here as needed.
