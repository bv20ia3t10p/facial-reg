"""
Routes package for the BioEmo API.
Contains authentication, user management, analytics, and verification endpoints.
"""

from fastapi import APIRouter
from . import auth, users, analytics, verification, client_users

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(verification.router, prefix="/verification", tags=["verification"])
api_router.include_router(client_users.router, prefix="/client-users", tags=["client-users"])