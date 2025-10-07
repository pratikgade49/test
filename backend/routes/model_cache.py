from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User
from auth import get_current_user
from model_persistence import ModelPersistenceManager, SavedModel, ModelAccuracyHistory

router = APIRouter()

@router.get("/accuracy_history/{config_hash}")
async def get_accuracy_history(
    config_hash: str,
    days_back: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get accuracy history for a configuration"""
    try:
        history = ModelPersistenceManager.get_accuracy_history(db, config_hash, days_back)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting accuracy history: {str(e)}")

@router.get("/model_cache_info")
async def get_model_cache_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get information about cached models"""
    try:
        models = db.query(SavedModel).order_by(SavedModel.last_used.desc()).limit(50).all()

        result = []
        for model in models:
            result.append({
                'model_hash': model.model_hash,
                'algorithm': model.algorithm,
                'accuracy': model.accuracy,
                'created_at': model.created_at.isoformat(),
                'last_used': model.last_used.isoformat(),
                'use_count': model.use_count
            })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model cache info: {str(e)}")

@router.post("/clear_all_model_cache")
async def clear_all_model_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear ALL cached models immediately"""
    try:
        before_count = db.query(SavedModel).count()

        db.query(SavedModel).delete()
        db.query(ModelAccuracyHistory).delete()
        db.commit()

        after_count = db.query(SavedModel).count()
        cleared_count = before_count - after_count

        return {
            "message": f"All model cache cleared successfully",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing all model cache: {str(e)}")

@router.post("/clear_model_cache")
async def clear_model_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear old cached models"""
    try:
        before_count = db.query(SavedModel).count()
        ModelPersistenceManager.cleanup_old_models(db, days_old=7, max_models=50)
        after_count = db.query(SavedModel).count()
        cleared_count = before_count - after_count

        return {
            "message": f"Model cache cleanup completed",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing model cache: {str(e)}")