from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User
from auth import get_current_user
from scheduler import (
    ScheduledForecast, ForecastExecution, ScheduleFrequency, ScheduleStatus,
    get_scheduler_status
)
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ScheduledForecastCreate(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: dict
    frequency: str
    start_date: datetime
    end_date: Optional[datetime] = None

class ScheduledForecastUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None

class ScheduledForecastResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str]
    forecast_config: dict
    frequency: str
    start_date: datetime
    end_date: Optional[datetime]
    next_run: datetime
    last_run: Optional[datetime]
    status: str
    run_count: int
    success_count: int
    failure_count: int
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime

class ForecastExecutionResponse(BaseModel):
    id: int
    scheduled_forecast_id: int
    execution_time: datetime
    status: str
    duration_seconds: Optional[int]
    result_summary: Optional[dict]
    error_message: Optional[str]
    created_at: datetime

@router.post("", response_model=ScheduledForecastResponse)
async def create_scheduled_forecast(
    request: ScheduledForecastCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new scheduled forecast"""
    try:
        try:
            frequency_enum = ScheduleFrequency(request.frequency.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid frequency. Must be one of: {[f.value for f in ScheduleFrequency]}"
            )
        
        next_run = request.start_date
        if request.start_date <= datetime.utcnow():
            if frequency_enum == ScheduleFrequency.DAILY:
                next_run = datetime.utcnow() + timedelta(days=1)
            elif frequency_enum == ScheduleFrequency.WEEKLY:
                next_run = datetime.utcnow() + timedelta(weeks=1)
            elif frequency_enum == ScheduleFrequency.MONTHLY:
                now = datetime.utcnow()
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1)
                else:
                    next_run = now.replace(month=now.month + 1)
        
        scheduled_forecast = ScheduledForecast(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            forecast_config=json.dumps(request.forecast_config),
            frequency=frequency_enum,
            start_date=request.start_date,
            end_date=request.end_date,
            next_run=next_run,
            status=ScheduleStatus.ACTIVE
        )
        
        db.add(scheduled_forecast)
        db.commit()
        db.refresh(scheduled_forecast)
        
        return ScheduledForecastResponse(
            id=scheduled_forecast.id,
            user_id=scheduled_forecast.user_id,
            name=scheduled_forecast.name,
            description=scheduled_forecast.description,
            forecast_config=json.loads(scheduled_forecast.forecast_config),
            frequency=scheduled_forecast.frequency.value,
            start_date=scheduled_forecast.start_date,
            end_date=scheduled_forecast.end_date,
            next_run=scheduled_forecast.next_run,
            last_run=scheduled_forecast.last_run,
            status=scheduled_forecast.status.value,
            run_count=scheduled_forecast.run_count,
            success_count=scheduled_forecast.success_count,
            failure_count=scheduled_forecast.failure_count,
            last_error=scheduled_forecast.last_error,
            created_at=scheduled_forecast.created_at,
            updated_at=scheduled_forecast.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error creating scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=List[ScheduledForecastResponse])
async def get_scheduled_forecasts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all scheduled forecasts for the current user"""
    try:
        scheduled_forecasts = db.query(ScheduledForecast).filter(
            ScheduledForecast.user_id == current_user.id
        ).order_by(ScheduledForecast.created_at.desc()).all()
        
        return [
            ScheduledForecastResponse(
                id=sf.id,
                user_id=sf.user_id,
                name=sf.name,
                description=sf.description,
                forecast_config=json.loads(sf.forecast_config),
                frequency=sf.frequency.value,
                start_date=sf.start_date,
                end_date=sf.end_date,
                next_run=sf.next_run,
                last_run=sf.last_run,
                status=sf.status.value,
                run_count=sf.run_count,
                success_count=sf.success_count,
                failure_count=sf.failure_count,
                last_error=sf.last_error,
                created_at=sf.created_at,
                updated_at=sf.updated_at
            )
            for sf in scheduled_forecasts
        ]
        
    except Exception as e:
        logger.error(f"Error fetching scheduled forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{forecast_id}", response_model=ScheduledForecastResponse)
async def get_scheduled_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific scheduled forecast"""
    try:
        scheduled_forecast = db.query(ScheduledForecast).filter(
            ScheduledForecast.id == forecast_id,
            ScheduledForecast.user_id == current_user.id
        ).first()
        
        if not scheduled_forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        return ScheduledForecastResponse(
            id=scheduled_forecast.id,
            user_id=scheduled_forecast.user_id,
            name=scheduled_forecast.name,
            description=scheduled_forecast.description,
            forecast_config=json.loads(scheduled_forecast.forecast_config),
            frequency=scheduled_forecast.frequency.value,
            start_date=scheduled_forecast.start_date,
            end_date=scheduled_forecast.end_date,
            next_run=scheduled_forecast.next_run,
            last_run=scheduled_forecast.last_run,
            status=scheduled_forecast.status.value,
            run_count=scheduled_forecast.run_count,
            success_count=scheduled_forecast.success_count,
            failure_count=scheduled_forecast.failure_count,
            last_error=scheduled_forecast.last_error,
            created_at=scheduled_forecast.created_at,
            updated_at=scheduled_forecast.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{forecast_id}", response_model=ScheduledForecastResponse)
async def update_scheduled_forecast(
    forecast_id: int,
    request: ScheduledForecastUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a scheduled forecast"""
    try:
        scheduled_forecast = db.query(ScheduledForecast).filter(
            ScheduledForecast.id == forecast_id,
            ScheduledForecast.user_id == current_user.id
        ).first()
        
        if not scheduled_forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        if request.name is not None:
            scheduled_forecast.name = request.name
        if request.description is not None:
            scheduled_forecast.description = request.description
        if request.frequency is not None:
            try:
                scheduled_forecast.frequency = ScheduleFrequency(request.frequency.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid frequency. Must be one of: {[f.value for f in ScheduleFrequency]}"
                )
        if request.start_date is not None:
            scheduled_forecast.start_date = request.start_date
        if request.end_date is not None:
            scheduled_forecast.end_date = request.end_date
        if request.status is not None:
            try:
                scheduled_forecast.status = ScheduleStatus(request.status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status. Must be one of: {[s.value for s in ScheduleStatus]}"
                )
        
        scheduled_forecast.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(scheduled_forecast)
        
        return ScheduledForecastResponse(
            id=scheduled_forecast.id,
            user_id=scheduled_forecast.user_id,
            name=scheduled_forecast.name,
            description=scheduled_forecast.description,
            forecast_config=json.loads(scheduled_forecast.forecast_config),
            frequency=scheduled_forecast.frequency.value,
            start_date=scheduled_forecast.start_date,
            end_date=scheduled_forecast.end_date,
            next_run=scheduled_forecast.next_run,
            last_run=scheduled_forecast.last_run,
            status=scheduled_forecast.status.value,
            run_count=scheduled_forecast.run_count,
            success_count=scheduled_forecast.success_count,
            failure_count=scheduled_forecast.failure_count,
            last_error=scheduled_forecast.last_error,
            created_at=scheduled_forecast.created_at,
            updated_at=scheduled_forecast.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{forecast_id}", response_model=dict)
async def delete_scheduled_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a scheduled forecast"""
    try:
        scheduled_forecast = db.query(ScheduledForecast).filter(
            ScheduledForecast.id == forecast_id,
            ScheduledForecast.user_id == current_user.id
        ).first()
        
        if not scheduled_forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        db.query(ForecastExecution).filter(
            ForecastExecution.scheduled_forecast_id == forecast_id
        ).delete()
        
        db.delete(scheduled_forecast)
        db.commit()
        
        return {"message": "Scheduled forecast deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{forecast_id}/executions", response_model=List[ForecastExecutionResponse])
async def get_forecast_executions(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get execution history for a scheduled forecast"""
    try:
        scheduled_forecast = db.query(ScheduledForecast).filter(
            ScheduledForecast.id == forecast_id,
            ScheduledForecast.user_id == current_user.id
        ).first()
        
        if not scheduled_forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        executions = db.query(ForecastExecution).filter(
            ForecastExecution.scheduled_forecast_id == forecast_id
        ).order_by(ForecastExecution.execution_time.desc()).all()
        
        return [
            ForecastExecutionResponse(
                id=execution.id,
                scheduled_forecast_id=execution.scheduled_forecast_id,
                execution_time=execution.execution_time,
                status=execution.status,
                duration_seconds=execution.duration_seconds,
                result_summary=json.loads(execution.result_summary) if execution.result_summary else None,
                error_message=execution.error_message,
                created_at=execution.created_at
            )
            for execution in executions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduler/status", response_model=dict)
async def get_scheduler_status_endpoint():
    """Get the current scheduler status"""
    return get_scheduler_status()