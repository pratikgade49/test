from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User, SavedForecastResult, DimensionManager
from auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import json
from datetime import datetime

router = APIRouter()

class SavedForecastRequest(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: dict
    forecast_data: Union[dict, Any]

class SavedForecastResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str]
    forecast_config: dict
    forecast_data: Union[dict, Any]
    created_at: str
    updated_at: str

@router.get("", response_model=List[dict])
async def get_saved_forecasts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all saved forecasts for the current user"""
    try:
        saved_forecasts = db.query(SavedForecastResult).filter(
            SavedForecastResult.user_id == current_user.id
        ).order_by(SavedForecastResult.created_at.desc()).all()

        result = []
        for forecast in saved_forecasts:
            try:
                forecast_config_dict = json.loads(forecast.forecast_config)
                forecast_data_dict = json.loads(forecast.forecast_data)

                # Convert IDs back to names for display
                if 'selectedItem' in forecast_config_dict and forecast_config_dict['selectedItem'] is None and 'selected_item_id' in forecast_config_dict:
                    forecast_config_dict['selectedItem'] = DimensionManager.get_dimension_name(
                        db, forecast_config_dict['forecastBy'], forecast_config_dict['selected_item_id']
                    )
                if 'selectedProduct' in forecast_config_dict and forecast_config_dict['selectedProduct'] is None and 'selected_product_id' in forecast_config_dict:
                    forecast_config_dict['selectedProduct'] = DimensionManager.get_dimension_name(
                        db, 'product', forecast_config_dict['selected_product_id']
                    )
                if 'selectedCustomer' in forecast_config_dict and forecast_config_dict['selectedCustomer'] is None and 'selected_customer_id' in forecast_config_dict:
                    forecast_config_dict['selectedCustomer'] = DimensionManager.get_dimension_name(
                        db, 'customer', forecast_config_dict['selected_customer_id']
                    )
                if 'selectedLocation' in forecast_config_dict and forecast_config_dict['selectedLocation'] is None and 'selected_location_id' in forecast_config_dict:
                    forecast_config_dict['selectedLocation'] = DimensionManager.get_dimension_name(
                        db, 'location', forecast_config_dict['selected_location_id']
                    )
                
                # Handle multi-select items
                if 'selectedItems' in forecast_config_dict and forecast_config_dict['selectedItems'] is None and 'selected_items_ids' in forecast_config_dict and forecast_config_dict['selected_items_ids']:
                    selected_ids = json.loads(forecast_config_dict['selected_items_ids'])
                    selected_names = []
                    for item_id in selected_ids:
                        selected_names.append(DimensionManager.get_dimension_name(
                            db, forecast_config_dict['forecastBy'], item_id
                        ))
                    forecast_config_dict['selectedItems'] = selected_names

                # Handle combination names in forecast_data
                if 'results' in forecast_data_dict and forecast_data_dict['results']:
                    for res in forecast_data_dict['results']:
                        if 'combination' in res and res['combination']:
                            if 'product_id' in res['combination'] and res['combination']['product_id'] is not None:
                                res['combination']['product'] = DimensionManager.get_dimension_name(
                                    db, 'product', res['combination']['product_id']
                                )
                            if 'customer_id' in res['combination'] and res['combination']['customer_id'] is not None:
                                res['combination']['customer'] = DimensionManager.get_dimension_name(
                                    db, 'customer', res['combination']['customer_id']
                                )
                            if 'location_id' in res['combination'] and res['combination']['location_id'] is not None:
                                res['combination']['location'] = DimensionManager.get_dimension_name(
                                    db, 'location', res['combination']['location_id']
                                )
                
                # Handle best/worst combination names
                if 'summary' in forecast_data_dict and forecast_data_dict['summary']:
                    if 'bestCombination' in forecast_data_dict['summary'] and forecast_data_dict['summary']['bestCombination']['combination']:
                        best_combo = forecast_data_dict['summary']['bestCombination']['combination']
                        if 'product_id' in best_combo and best_combo['product_id'] is not None:
                            best_combo['product'] = DimensionManager.get_dimension_name(
                                db, 'product', best_combo['product_id']
                            )
                        if 'customer_id' in best_combo and best_combo['customer_id'] is not None:
                            best_combo['customer'] = DimensionManager.get_dimension_name(
                                db, 'customer', best_combo['customer_id']
                            )
                        if 'location_id' in best_combo and best_combo['location_id'] is not None:
                            best_combo['location'] = DimensionManager.get_dimension_name(
                                db, 'location', best_combo['location_id']
                            )
                    
                    if 'worstCombination' in forecast_data_dict['summary'] and forecast_data_dict['summary']['worstCombination']['combination']:
                        worst_combo = forecast_data_dict['summary']['worstCombination']['combination']
                        if 'product_id' in worst_combo and worst_combo['product_id'] is not None:
                            worst_combo['product'] = DimensionManager.get_dimension_name(
                                db, 'product', worst_combo['product_id']
                            )
                        if 'customer_id' in worst_combo and worst_combo['customer_id'] is not None:
                            worst_combo['customer'] = DimensionManager.get_dimension_name(
                                db, 'customer', worst_combo['customer_id']
                            )
                        if 'location_id' in worst_combo and worst_combo['location_id'] is not None:
                            worst_combo['location'] = DimensionManager.get_dimension_name(
                                db, 'location', worst_combo['location_id']
                            )

                result.append({
                    'id': forecast.id,
                    'user_id': forecast.user_id,
                    'name': forecast.name,
                    'description': forecast.description,
                    'forecast_config': forecast_config_dict,
                    'forecast_data': forecast_data_dict,
                    'created_at': forecast.created_at.isoformat(),
                    'updated_at': forecast.updated_at.isoformat()
                })
            except Exception as e:
                print(f"Error parsing saved forecast {forecast.id}: {e}")
                continue

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting saved forecasts: {str(e)}")

@router.post("", response_model=dict)
async def save_forecast(
    request: SavedForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a forecast result"""
    try:
        config_dict = request.forecast_config.copy()
        
        # Convert names to IDs in forecast_config
        if config_dict.get('selectedItem'):
            config_dict['selected_item_id'] = DimensionManager.get_dimension_id(
                db, config_dict['forecastBy'], config_dict['selectedItem']
            )
            del config_dict['selectedItem']
        if config_dict.get('selectedProduct'):
            config_dict['selected_product_id'] = DimensionManager.get_dimension_id(
                db, 'product', config_dict['selectedProduct']
            )
            del config_dict['selectedProduct']
        if config_dict.get('selectedCustomer'):
            config_dict['selected_customer_id'] = DimensionManager.get_dimension_id(
                db, 'customer', config_dict['selectedCustomer']
            )
            del config_dict['selectedCustomer']
        if config_dict.get('selectedLocation'):
            config_dict['selected_location_id'] = DimensionManager.get_dimension_id(
                db, 'location', config_dict['selectedLocation']
            )
            del config_dict['selectedLocation']
        
        # Handle multi-select items
        if config_dict.get('selectedItems'):
            selected_ids = []
            for item_name in config_dict['selectedItems']:
                item_id = DimensionManager.get_dimension_id(
                    db, config_dict['forecastBy'], item_name
                )
                if item_id is not None:
                    selected_ids.append(item_id)
            config_dict['selected_items_ids'] = json.dumps(selected_ids)
            del config_dict['selectedItems']

        # Convert forecast_data to dict if needed
        if hasattr(request.forecast_data, 'dict'):
            forecast_data_dict = request.forecast_data.dict()
        else:
            forecast_data_dict = request.forecast_data

        # Convert names to IDs in forecast_data
        if 'results' in forecast_data_dict and forecast_data_dict['results']:
            for res in forecast_data_dict['results']:
                if 'combination' in res and res['combination']:
                    if 'product' in res['combination'] and res['combination']['product'] is not None:
                        res['combination']['product_id'] = DimensionManager.get_dimension_id(
                            db, 'product', res['combination']['product']
                        )
                        del res['combination']['product']
                    if 'customer' in res['combination'] and res['combination']['customer'] is not None:
                        res['combination']['customer_id'] = DimensionManager.get_dimension_id(
                            db, 'customer', res['combination']['customer']
                        )
                        del res['combination']['customer']
                    if 'location' in res['combination'] and res['combination']['location'] is not None:
                        res['combination']['location_id'] = DimensionManager.get_dimension_id(
                            db, 'location', res['combination']['location']
                        )
                        del res['combination']['location']

        saved_forecast = SavedForecastResult(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            forecast_config=json.dumps(config_dict),
            forecast_data=json.dumps(forecast_data_dict)
        )

        db.add(saved_forecast)
        db.commit()
        db.refresh(saved_forecast)

        return {
            "message": "Forecast saved successfully",
            "id": saved_forecast.id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving forecast: {str(e)}")

@router.delete("/{forecast_id}", response_model=dict)
async def delete_saved_forecast(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved forecast (only if it belongs to the current user)"""
    try:
        saved_forecast = db.query(SavedForecastResult).filter(
            SavedForecastResult.id == forecast_id,
            SavedForecastResult.user_id == current_user.id
        ).first()

        if not saved_forecast:
            raise HTTPException(
                status_code=404, 
                detail="Saved forecast not found or you don't have permission to delete it"
            )

        db.delete(saved_forecast)
        db.commit()

        return {"message": "Saved forecast deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting saved forecast: {str(e)}")