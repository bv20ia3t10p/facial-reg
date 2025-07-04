"""
Verification request routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Body
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import random
import base64
import json
from pydantic import BaseModel

from ..db.database import get_db, User, VerificationRequest, create_verification_request
from ..utils.security import get_current_user, verify_password

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Store OTPs temporarily (in a real system, this would be in a database with expiration)
otp_store = {}

class OTPRequest(BaseModel):
    user_id: str
    reason: str
    additional_notes: Optional[str] = None
    capturedImage: str
    confidence: float

class VerifyOTPRequest(BaseModel):
    user_id: str
    otp: str

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
        
        result = []
        for req in requests:
            # Parse request data from JSON
            request_details = {}
            if req.request_data is not None:
                try:
                    request_details = json.loads(str(req.request_data))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse request_data for request {req.id}")
                    request_details = {}
            
            # Handle image data - ensure it's a valid string
            image_data = request_details.get("captured_image")
            if image_data and isinstance(image_data, bytes):
                try:
                    # Try to decode as UTF-8 first (in case it's already a string stored as bytes)
                    image_data = image_data.decode('utf-8')
                except UnicodeDecodeError:
                    # If that fails, it's binary data, so encode to base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
            
            result.append({
                "id": str(req.id),
                "employeeId": str(req.user_id),
                "reason": request_details.get("reason", ""),
                "additionalNotes": request_details.get("additional_notes", ""),
                "capturedImage": image_data,
                "confidence": request_details.get("confidence", 0.0),
                "status": req.status,
                "submittedAt": req.created_at.isoformat(),
                "processedAt": req.processed_at.isoformat() if req.processed_at is not None else None,
                "user": {
                    "id": str(req.user.id),
                    "name": req.user.name,
                    "department": req.user.department
                } if req.user else None
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get verification requests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/requests")
async def create_verification_request_endpoint(
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
        # Store all request details as JSON in request_data column
        request_details = {
            "reason": reason,
            "additional_notes": additional_notes,
            "captured_image": base64.b64encode(image_data).decode('utf-8'),
            "confidence": 0.0  # Will be updated when processed
        }
        
        request = VerificationRequest(
            user_id=current_user.id,
            status="pending",
            request_data=json.dumps(request_details)
        )
        
        db.add(request)
        db.commit()
        db.refresh(request)
        
        return {
            "id": str(request.id),
            "employeeId": str(request.user_id),
            "reason": reason,
            "additionalNotes": additional_notes,
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
    request_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a verification request status"""
    try:
        # Get the status from the request body
        request_status = request_data.get("status")
        if not request_status:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status is required"
            )
            
        # Get the request
        request = db.query(VerificationRequest).filter(
            VerificationRequest.id == request_id
        ).first()
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Verification request not found"
            )
        
        # Update status using setattr to work with SQLAlchemy
        setattr(request, 'status', request_status)
        setattr(request, 'processed_at', datetime.utcnow())
        
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

@router.post("/request-otp")
async def request_otp(
    request_data: OTPRequest,
    db: Session = Depends(get_db)
):
    """Request OTP verification for low confidence authentication"""
    try:
        # Check if user exists
        user = db.query(User).filter(User.id == request_data.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate OTP (6-digit number)
        otp = ''.join(random.choices('0123456789', k=6))
        
        # Store OTP with user ID (in a real system, this would be stored in a database with expiration)
        otp_store[request_data.user_id] = {
            'otp': otp,
            'created_at': datetime.utcnow()
        }
        
        # Create verification request in the database
        # Store all request details as JSON in request_data column
        request_details = {
            "reason": request_data.reason,
            "additional_notes": request_data.additional_notes,
            "captured_image": request_data.capturedImage,  # Store as base64 string
            "confidence": request_data.confidence
        }
        
        verification_data = {
            "user_id": user.id,  # Use the user.id to ensure it's valid
            "status": "pending",
            "request_data": json.dumps(request_details)
        }
        
        try:
            # Create the verification request
            verification_request = create_verification_request(db, verification_data)
            if not verification_request:
                logger.error(f"Failed to create verification request for user {user.id}")
                # Continue even if request creation fails - we'll just return the OTP
        except Exception as req_error:
            logger.error(f"Error creating verification request: {str(req_error)}")
            # Continue - don't block OTP generation due to request failure
        
        logger.info(f"Generated OTP for user: {user.id}")
        
        # In a production system, we would send the OTP via email/SMS
        # For development, we'll return it in the response
        return {
            "success": True,
            "message": "OTP generated and sent",
            "otp": otp  # Only for development/testing
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to request OTP: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request OTP: {str(e)}"
        )

@router.post("/verify-otp")
async def verify_otp(
    verify_data: VerifyOTPRequest,
    db: Session = Depends(get_db)
):
    """Verify OTP for low confidence authentication"""
    try:
        user_id = verify_data.user_id
        otp = verify_data.otp
        
        # Check if OTP exists for user
        if user_id not in otp_store:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No OTP request found for this user"
            )
        
        stored_otp_data = otp_store[user_id]
        
        # Check if OTP is correct
        if stored_otp_data['otp'] != otp:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OTP"
            )
        
        # Check if OTP is expired (in a real system)
        # For now, we'll just mark it as verified
        stored_otp_data['verified'] = True
        
        # Get user data to return
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update all pending verification requests for this user
        pending_requests = db.query(VerificationRequest).filter(
            VerificationRequest.user_id == user_id,
            VerificationRequest.status == 'pending'
        ).all()
        
        for request in pending_requests:
            setattr(request, 'status', 'approved')
            setattr(request, 'processed_at', datetime.utcnow())
        
        db.commit()
        
        return {
            "success": True,
            "message": "OTP verified successfully",
            "user_id": user_id,
            "user": {
                "id": str(user.id),
                "name": user.name,
                "email": user.email,
                "department": user.department,
                "role": user.role
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to verify OTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/verify-credentials")
async def verify_credentials(
    credentials: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Verify user credentials for low confidence authentication"""
    try:
        user_id = credentials.get('user_id')
        password = credentials.get('password')
        
        if not user_id or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID and password are required"
            )
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify password (for demo, we'll accept "demo" as password)
        # Access password_hash as a string attribute
        user_password_hash = str(user.password_hash) if user.password_hash is not None else ""
        if password != "demo" and not verify_password(password, user_password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Generate OTP for further verification
        otp = ''.join(random.choices('0123456789', k=6))
        
        # Store OTP with user ID
        otp_store[user_id] = {
            'otp': otp,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow().isoformat()),
            'verified': False
        }
        
        # In a real system, send OTP to user via email/SMS
        logger.info(f"OTP generated for user {user_id} after credential verification: {otp}")
        
        return {
            "success": True,
            "message": "Credentials verified, OTP sent",
            "user_id": user_id,
            "otp": otp  # Only for testing purposes
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to verify credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 