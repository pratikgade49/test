from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User, ForecastConfiguration, DimensionManager
from auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
from sqlalchemy import and_

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

class SaveConfigRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: ForecastConfig

class ConfigurationResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    config: ForecastConfig
    createdAt: str
    updatedAt: str

@router.get("", response_model=dict)
async def get_configurations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all saved configurations"""
    try:
        configs = db.query(ForecastConfiguration).order_by(
            ForecastConfiguration.updated_at.desc()
        ).all()
        
        result = []
        for config in configs:
            selected_item_name = None
            if config.selected_item_id:
                if config.forecast_by == 'product':
                    selected_item_name = DimensionManager.get_dimension_name(db, 'product', config.selected_item_id)
                elif config.forecast_by == 'customer':
                    selected_item_name = DimensionManager.get_dimension_name(db, 'customer', config.selected_item_id)
                elif config.forecast_by == 'location':
                    selected_item_name = DimensionManager.get_dimension_name(db, 'location', config.selected_item_id)

            selected_product_name = DimensionManager.get_dimension_name(db, 'product', config.selected_product_id) if config.selected_product_id else None
            selected_customer_name = DimensionManager.get_dimension_name(db, 'customer', config.selected_customer_id) if config.selected_customer_id else None
            selected_location_name = DimensionManager.get_dimension_name(db, 'location', config.selected_location_id) if config.selected_location_id else None

            selected_items_names = []
            if config.selected_items_ids:
                selected_ids = json.loads(config.selected_items_ids)
                for item_id in selected_ids:
                    if config.forecast_by == 'product':
                        selected_items_names.append(DimensionManager.get_dimension_name(db, 'product', item_id))
                    elif config.forecast_by == 'customer':
                        selected_items_names.append(DimensionManager.get_dimension_name(db, 'customer', item_id))
                    elif config.forecast_by == 'location':
                        selected_items_names.append(DimensionManager.get_dimension_name(db, 'location', item_id))

            result.append(ConfigurationResponse(
                id=config.id,
                name=config.name,
                description=config.description,
                config=ForecastConfig(
                    forecastBy=config.forecast_by,
                    selectedItem=selected_item_name,
                    selectedProduct=selected_product_name,
                    selectedCustomer=selected_customer_name,
                    selectedLocation=selected_location_name,
                    selectedItems=selected_items_names if selected_items_names else None,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historic_period,
                    forecastPeriod=config.forecast_period
                ),
                createdAt=config.created_at.isoformat(),
                updatedAt=config.updated_at.isoformat()
            ))
        
        return {"configurations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configurations: {str(e)}")

@router.post("", response_model=dict)
async def save_configuration(
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a new configuration"""
    try:
        existing = db.query(ForecastConfiguration).filter(
            ForecastConfiguration.name == request.name
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Configuration with name '{request.name}' already exists"
            )
        
        selected_item_id = None
        if request.config.selectedItem:
            selected_item_id = DimensionManager.get_dimension_id(
                db, request.config.forecastBy, request.config.selectedItem
            )
        
        selected_product_id = None
        if request.config.selectedProduct:
            selected_product_id = DimensionManager.get_dimension_id(
                db, 'product', request.config.selectedProduct
            )
        
        selected_customer_id = None
        if request.config.selectedCustomer:
            selected_customer_id = DimensionManager.get_dimension_id(
                db, 'customer', request.config.selectedCustomer
            )
        
        selected_location_id = None
        if request.config.selectedLocation:
            selected_location_id = DimensionManager.get_dimension_id(
                db, 'location', request.config.selectedLocation
            )

        selected_items_ids_json = None
        if request.config.selectedItems:
            selected_ids = []
            for item_name in request.config.selectedItems:
                item_id = DimensionManager.get_dimension_id(
                    db, request.config.forecastBy, item_name
                )
                if item_id is not None:
                    selected_ids.append(item_id)
            selected_items_ids_json = json.dumps(selected_ids)

        config = ForecastConfiguration(
            name=request.name,
            description=request.description,
            forecast_by=request.config.forecastBy,
            selected_item_id=selected_item_id,
            selected_product_id=selected_product_id,
            selected_customer_id=selected_customer_id,
            selected_location_id=selected_location_id,
            selected_items_ids=selected_items_ids_json,
            algorithm=request.config.algorithm,
            interval=request.config.interval,
            historic_period=request.config.historicPeriod,
            forecast_period=request.config.forecastPeriod
        )
        
        db.add(config)
        db.commit()
        db.refresh(config)
        
        return {
            "message": "Configuration saved successfully",
            "id": config.id,
            "name": config.name
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

@router.get("/{config_id}", response_model=ConfigurationResponse)
async def get_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific configuration by ID"""
    try:
        config = db.query(ForecastConfiguration).filter(
            ForecastConfiguration.id == config_id
        ).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        selected_item_name = None
        if config.selected_item_id:
            if config.forecast_by == 'product':
                selected_item_name = DimensionManager.get_dimension_name(db, 'product', config.selected_item_id)
            elif config.forecast_by == 'customer':
                selected_item_name = DimensionManager.get_dimension_name(db, 'customer', config.selected_item_id)
            elif config.forecast_by == 'location':
                selected_item_name = DimensionManager.get_dimension_name(db, 'location', config.selected_item_id)

        selected_product_name = DimensionManager.get_dimension_name(db, 'product', config.selected_product_id) if config.selected_product_id else None
        selected_customer_name = DimensionManager.get_dimension_name(db, 'customer', config.selected_customer_id) if config.selected_customer_id else None
        selected_location_name = DimensionManager.get_dimension_name(db, 'location', config.selected_location_id) if config.selected_location_id else None

        selected_items_names = []
        if config.selected_items_ids:
            selected_ids = json.loads(config.selected_items_ids)
            for item_id in selected_ids:
                if config.forecast_by == 'product':
                    selected_items_names.append(DimensionManager.get_dimension_name(db, 'product', item_id))
                elif config.forecast_by == 'customer':
                    selected_items_names.append(DimensionManager.get_dimension_name(db, 'customer', item_id))
                elif config.forecast_by == 'location':
                    selected_items_names.append(DimensionManager.get_dimension_name(db, 'location', item_id))

        return ConfigurationResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            config=ForecastConfig(
                forecastBy=config.forecast_by,
                selectedItem=selected_item_name,
                selectedProduct=selected_product_name,
                selectedCustomer=selected_customer_name,
                selectedLocation=selected_location_name,
                selectedItems=selected_items_names if selected_items_names else None,
                algorithm=config.algorithm,
                interval=config.interval,
                historicPeriod=config.historic_period,
                forecastPeriod=config.forecast_period
            ),
            createdAt=config.created_at.isoformat(),
            updatedAt=config.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@router.put("/{config_id}", response_model=dict)
async def update_configuration(
    config_id: int,
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update an existing configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(
            ForecastConfiguration.id == config_id
        ).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        if request.name != config.name:
            existing = db.query(ForecastConfiguration).filter(
                and_(
                    ForecastConfiguration.name == request.name, 
                    ForecastConfiguration.id != config_id
                )
            ).first()
            if existing:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Configuration with name '{request.name}' already exists"
                )
        
        selected_item_id = None
        if request.config.selectedItem:
            selected_item_id = DimensionManager.get_dimension_id(
                db, request.config.forecastBy, request.config.selectedItem
            )
        
        selected_product_id = None
        if request.config.selectedProduct:
            selected_product_id = DimensionManager.get_dimension_id(
                db, 'product', request.config.selectedProduct
            )
        
        selected_customer_id = None
        if request.config.selectedCustomer:
            selected_customer_id = DimensionManager.get_dimension_id(
                db, 'customer', request.config.selectedCustomer
            )
        
        selected_location_id = None
        if request.config.selectedLocation:
            selected_location_id = DimensionManager.get_dimension_id(
                db, 'location', request.config.selectedLocation
            )

        selected_items_ids_json = None
        if request.config.selectedItems:
            selected_ids = []
            for item_name in request.config.selectedItems:
                item_id = DimensionManager.get_dimension_id(
                    db, request.config.forecastBy, item_name
                )
                if item_id is not None:
                    selected_ids.append(item_id)
            selected_items_ids_json = json.dumps(selected_ids)

        config.name = request.name
        config.description = request.description
        config.forecast_by = request.config.forecastBy
        config.selected_item_id = selected_item_id
        config.selected_product_id = selected_product_id
        config.selected_customer_id = selected_customer_id
        config.selected_location_id = selected_location_id
        config.selected_items_ids = selected_items_ids_json
        config.algorithm = request.config.algorithm
        config.interval = request.config.interval
        config.historic_period = request.config.historicPeriod
        config.forecast_period = request.config.forecastPeriod
        config.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "message": "Configuration updated successfully",
            "id": config.id,
            "name": config.name
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@router.delete("/{config_id}", response_model=dict)
async def delete_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(
            ForecastConfiguration.id == config_id
        ).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        db.delete(config)
        db.commit()
        
        return {"message": "Configuration deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting configuration: {str(e)}")