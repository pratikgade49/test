from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db, User
from auth import get_current_user
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_approved: Optional[bool] = None
    is_admin: Optional[bool] = None
    created_at: str

class AdminSetActiveRequest(BaseModel):
    is_active: bool

def require_admin(user: User):
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """List all users (admin only)"""
    require_admin(current_user)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            is_active=bool(u.is_active),
            is_approved=bool(u.is_approved),
            created_at=u.created_at.isoformat()
        ) for u in users
    ]

@router.post("/users/{user_id}/approve", response_model=UserResponse)
async def approve_user(
    user_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Approve a pending user (admin only)"""
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    user.is_approved = True
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=bool(user.is_active),
        is_approved=bool(user.is_approved),
        created_at=user.created_at.isoformat()
    )

@router.post("/users/{user_id}/active", response_model=UserResponse)
async def set_user_active(
    user_id: int, 
    payload: AdminSetActiveRequest, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Activate/Deactivate a user (admin only)"""
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = bool(payload.is_active)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=bool(user.is_active),
        is_approved=bool(user.is_approved),
        is_admin=bool(user.is_admin),
        created_at=user.created_at.isoformat()
    )