"""
Verification request routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging

from api.src.db.database import get_db, User, VerificationRequest
from api.src.utils.security import get_current_user

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

@router.get("/requests")
async def get_verification_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all verification requests"""
    try:
        # Get all pending verification requests
        requests = db.query(VerificationRequest).order_by(
            VerificationRequest.created_at.desc()
        ).all()
        
        return [
            {
                "id": str(req.id),
                "employeeId": str(req.user_id),
                "reason": req.reason,
                "additionalNotes": req.additional_notes,
                "capturedImage": req.captured_image,
                "confidence": req.confidence,
                "status": req.status,
                "submittedAt": req.created_at.isoformat(),
                "processedAt": req.processed_at.isoformat() if req.processed_at else None,
                "user": {
                    "id": str(req.user.id),
                    "name": req.user.name,
                    "department": req.user.department
                } if req.user else None
            }
            for req in requests
        ]
        
    except Exception as e:
        logger.error(f"Failed to get verification requests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/requests")
async def create_verification_request(
    image: UploadFile = File(...),
    reason: str = Form(...),
    additional_notes: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new verification request"""
    try:
        # Read and store image
        image_data = await image.read()
        
        # Create verification request
        request = VerificationRequest(
            user_id=current_user.id,
            reason=reason,
            additional_notes=additional_notes,
            captured_image=image_data,
            confidence=0.0,  # Will be updated when processed
            status="pending"
        )
        
        db.add(request)
        db.commit()
        db.refresh(request)
        
        return {
            "id": str(request.id),
            "employeeId": str(request.user_id),
            "reason": request.reason,
            "additionalNotes": request.additional_notes,
            "status": request.status,
            "submittedAt": request.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create verification request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/requests/{request_id}")
async def update_verification_request(
    request_id: str,
    status: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a verification request status"""
    try:
        # Get the request
        request = db.query(VerificationRequest).filter(
            VerificationRequest.id == request_id
        ).first()
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Verification request not found"
            )
        
        # Update status
        request.status = status
        request.processed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(request)
        
        return {
            "id": str(request.id),
            "status": request.status,
            "processedAt": request.processed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update verification request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 