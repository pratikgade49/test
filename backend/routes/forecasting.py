from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User
from auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

router = APIRouter()

class ForecastConfig(BaseModel):
    forecastBy: str
    selectedItem: Optional[str] = None
    selectedProduct: Optional[str] = None
    selectedCustomer: Optional[str] = None
    selectedLocation: Optional[str] = None
    selectedProducts: Optional[List[str]] = None
    selectedCustomers: Optional[List[str]] = None
    selectedLocations: Optional[List[str]] = None
    selectedItems: Optional[List[str]] = None
    algorithm: str = "linear_regression"
    interval: str = "month"
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: bool = False
    advancedMode: bool = False
    externalFactors: Optional[List[str]] = None

@router.get("/algorithms")
async def get_algorithms():
    """Get available algorithms"""
    from main import ForecastingEngine
    return {"algorithms": ForecastingEngine.ALGORITHMS}

@router.post("/forecast")
def generate_forecast_endpoint(
    config: ForecastConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate forecast using data from database"""
    from main import ForecastingEngine
    from database import SavedForecastResult
    
    try:
        process_log = []
        process_log.append("=== Forecast Request Received ===")
        process_log.append(f"Multi-select mode: {config.multiSelect}")
        process_log.append(f"Advanced mode: {config.advancedMode}")

        if config.multiSelect:
            if config.advancedMode:
                process_log.append("Running advanced mode (precise combinations)")
                if not (config.selectedProducts and config.selectedCustomers and config.selectedLocations):
                    raise ValueError("Advanced mode requires selection of Products, Customers, and Locations")
                
                result = ForecastingEngine.generate_forecast_three_dimensions(
                    db, config, config.selectedProducts, config.selectedCustomers, 
                    config.selectedLocations, process_log
                )
            else:
                selected_dimensions = sum([
                    bool(config.selectedProducts),
                    bool(config.selectedCustomers),
                    bool(config.selectedLocations)
                ])

                if selected_dimensions >= 2:
                    result = ForecastingEngine.generate_multi_forecast(db, config, process_log)
                else:
                    raise ValueError("Multi-select mode requires at least 2 dimensions")

            # Auto-save multi-forecast
            _auto_save_forecast(db, current_user, config, result, "Multi-Forecast", process_log)
            return result

        elif config.selectedItems and len(config.selectedItems) > 1:
            process_log.append("Running simple multi-select mode")
            result = ForecastingEngine.generate_simple_multi_forecast(db, config, process_log)
            _auto_save_forecast(db, current_user, config, result, "Multi-Forecast", process_log)
            return result
        else:
            process_log.append("Running single selection mode")
            result = ForecastingEngine.generate_forecast(db, config, process_log)
            _auto_save_forecast(db, current_user, config, result, "Forecast", process_log)
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

def _auto_save_forecast(db, current_user, config, result, forecast_type, process_log):
    """Helper function to auto-save forecast results"""
    from database import SavedForecastResult
    
    try:
        auto_save_name = f"Auto-saved {forecast_type} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        auto_save_description = f"Automatically saved {forecast_type.lower()} result"

        existing = db.query(SavedForecastResult).filter(
            SavedForecastResult.user_id == current_user.id,
            SavedForecastResult.name == auto_save_name
        ).first()

        if existing:
            counter = 1
            while db.query(SavedForecastResult).filter(
                SavedForecastResult.user_id == current_user.id,
                SavedForecastResult.name == f"{auto_save_name} ({counter})"
            ).first():
                counter += 1
            auto_save_name = f"{auto_save_name} ({counter})"

        saved_forecast = SavedForecastResult(
            user_id=current_user.id,
            name=auto_save_name,
            description=auto_save_description,
            forecast_config=json.dumps(config.dict()),
            forecast_data=json.dumps(result.dict())
        )
        db.add(saved_forecast)
        db.commit()
        process_log.append(f"{forecast_type} automatically saved as '{auto_save_name}'")
    except Exception as save_error:
        process_log.append(f"Warning: Could not auto-save {forecast_type.lower()}: {str(save_error)}")

@router.post("/best_fit_recommendation")
async def get_best_fit_recommendation(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get algorithm recommendation based on previous best fit runs"""
    from model_persistence import ModelPersistenceManager
    from database import SavedModel
    from main import ForecastingEngine
    
    try:
        config_dict = request.get('config', {})
        config_for_hash = config_dict.copy()
        config_for_hash.pop('algorithm', None)

        config_hash = ModelPersistenceManager.generate_config_hash(config_for_hash)

        saved_models = db.query(SavedModel).filter(
            SavedModel.config_hash == config_hash,
            SavedModel.algorithm != 'best_fit'
        ).order_by(
            SavedModel.accuracy.desc(),
            SavedModel.last_used.desc()
        ).limit(5).all()

        if not saved_models:
            return {
                "recommended_algorithm": None,
                "message": "No previous best fit runs found for this configuration."
            }

        best_model = saved_models[0]
        
        if len(saved_models) > 1:
            top_algorithm_count = sum(1 for model in saved_models if model.algorithm == best_model.algorithm)
            confidence = (top_algorithm_count / len(saved_models)) * 100
        else:
            confidence = 75.0

        algorithm_display_name = ForecastingEngine.ALGORITHMS.get(
            best_model.algorithm, best_model.algorithm
        )

        return {
            "recommended_algorithm": best_model.algorithm,
            "confidence": round(confidence, 1),
            "last_accuracy": round(best_model.accuracy, 1),
            "last_run_date": best_model.last_used.isoformat(),
            "message": f"Based on previous runs, {algorithm_display_name} performed best with {best_model.accuracy:.1f}% accuracy."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendation: {str(e)}")