from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db, User, ForecastData, ExternalFactorData
from database import ProductDimension, CustomerDimension, LocationDimension, DimensionManager
from auth import get_current_user
from sqlalchemy import func, distinct
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
from datetime import datetime

router = APIRouter()

class DatabaseStats(BaseModel):
    totalRecords: int
    dateRange: Dict[str, str]
    uniqueProducts: int
    uniqueCustomers: int
    uniqueLocations: int

class DataViewRequest(BaseModel):
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    page: int = 1
    page_size: int = 50

class DataViewResponse(BaseModel):
    data: List[Dict[str, Any]]
    total_records: int
    page: int
    page_size: int
    total_pages: int

@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get database statistics"""
    try:
        total_records = db.query(func.count(ForecastData.id)).scalar()
        min_date = db.query(func.min(ForecastData.date)).scalar()
        max_date = db.query(func.max(ForecastData.date)).scalar()
        
        unique_products = db.query(func.count(distinct(ProductDimension.id))).scalar()
        unique_customers = db.query(func.count(distinct(CustomerDimension.id))).scalar()
        unique_locations = db.query(func.count(distinct(LocationDimension.id))).scalar()

        return DatabaseStats(
            totalRecords=total_records or 0,
            dateRange={
                "start": min_date.strftime('%Y-%m-%d') if min_date else "No data",
                "end": max_date.strftime('%Y-%m-%d') if max_date else "No data"
            },
            uniqueProducts=unique_products or 0,
            uniqueCustomers=unique_customers or 0,
            uniqueLocations=unique_locations or 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@router.get("/options")
async def get_database_options(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique values for dropdowns"""
    try:
        products = db.query(ProductDimension.product_name).distinct().all()
        customers = db.query(CustomerDimension.customer_name).distinct().all()
        locations = db.query(LocationDimension.location_name).distinct().all()

        return {
            "products": sorted([p[0] for p in products if p[0]]),
            "customers": sorted([c[0] for c in customers if c[0]]),
            "locations": sorted([l[0] for l in locations if l[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database options: {str(e)}")

@router.post("/filtered_options")
async def get_filtered_options(
    filters: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get filtered unique values based on selected filters"""
    try:
        query = db.query(ForecastData)

        selected_product_names = filters.get('selectedProducts', [])
        selected_customer_names = filters.get('selectedCustomers', [])
        selected_location_names = filters.get('selectedLocations', [])

        product_ids = [DimensionManager.get_dimension_id(db, 'product', name) 
                      for name in selected_product_names 
                      if DimensionManager.get_dimension_id(db, 'product', name) is not None]
        customer_ids = [DimensionManager.get_dimension_id(db, 'customer', name) 
                       for name in selected_customer_names 
                       if DimensionManager.get_dimension_id(db, 'customer', name) is not None]
        location_ids = [DimensionManager.get_dimension_id(db, 'location', name) 
                       for name in selected_location_names 
                       if DimensionManager.get_dimension_id(db, 'location', name) is not None]

        if product_ids:
            query = query.filter(ForecastData.product_id.in_(product_ids))
        if customer_ids:
            query = query.filter(ForecastData.customer_id.in_(customer_ids))
        if location_ids:
            query = query.filter(ForecastData.location_id.in_(location_ids))

        filtered_product_ids = query.with_entities(distinct(ForecastData.product_id)).filter(ForecastData.product_id.isnot(None)).all()
        filtered_customer_ids = query.with_entities(distinct(ForecastData.customer_id)).filter(ForecastData.customer_id.isnot(None)).all()
        filtered_location_ids = query.with_entities(distinct(ForecastData.location_id)).filter(ForecastData.location_id.isnot(None)).all()

        products = [DimensionManager.get_dimension_name(db, 'product', p[0]) for p in filtered_product_ids if p[0] is not None]
        customers = [DimensionManager.get_dimension_name(db, 'customer', c[0]) for c in filtered_customer_ids if c[0] is not None]
        locations = [DimensionManager.get_dimension_name(db, 'location', l[0]) for l in filtered_location_ids if l[0] is not None]

        return {
            "products": sorted(products),
            "customers": sorted(customers),
            "locations": sorted(locations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting filtered options: {str(e)}")

@router.post("/view", response_model=DataViewResponse)
async def view_database_data(
    request: DataViewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View database data with filters and pagination"""
    try:
        query = db.query(ForecastData, 
                        ProductDimension.product_group,
                        ProductDimension.product_hierarchy,
                        CustomerDimension.customer_group,
                        CustomerDimension.customer_region,
                        CustomerDimension.ship_to_party,
                        CustomerDimension.sold_to_party,
                        LocationDimension.location_region)\
                .outerjoin(ProductDimension, ForecastData.product_id == ProductDimension.id)\
                .outerjoin(CustomerDimension, ForecastData.customer_id == CustomerDimension.id)\
                .outerjoin(LocationDimension, ForecastData.location_id == LocationDimension.id)
        
        if request.product:
            product_id = DimensionManager.get_dimension_id(db, 'product', request.product)
            if product_id is not None:
                query = query.filter(ForecastData.product_id == product_id)
                
        if request.customer:
            customer_id = DimensionManager.get_dimension_id(db, 'customer', request.customer)
            if customer_id is not None:
                query = query.filter(ForecastData.customer_id == customer_id)
                
        if request.location:
            location_id = DimensionManager.get_dimension_id(db, 'location', request.location)
            if location_id is not None:
                query = query.filter(ForecastData.location_id == location_id)

        if request.start_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
            query = query.filter(ForecastData.date >= start_date)
            
        if request.end_date:
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
            query = query.filter(ForecastData.date <= end_date)
        
        total_records = query.count()
        offset = (request.page - 1) * request.page_size
        results = query.order_by(ForecastData.date.desc()).offset(offset).limit(request.page_size).all()
        
        data = []
        for record in results:
            forecast_data = record[0]
            data.append({
                'id': forecast_data.id,
                'product': DimensionManager.get_dimension_name(db, 'product', forecast_data.product_id) if forecast_data.product_id else None,
                'quantity': float(forecast_data.quantity) if forecast_data.quantity else 0,
                'product_group': record[1],
                'product_hierarchy': record[2],
                'location': DimensionManager.get_dimension_name(db, 'location', forecast_data.location_id) if forecast_data.location_id else None,
                'location_region': record[7],
                'customer': DimensionManager.get_dimension_name(db, 'customer', forecast_data.customer_id) if forecast_data.customer_id else None,
                'customer_group': record[3],
                'customer_region': record[4],
                'ship_to_party': record[5],
                'sold_to_party': record[6],
                'uom': forecast_data.uom,
                'date': forecast_data.date.strftime('%Y-%m-%d') if forecast_data.date else None,
                'unit_price': float(forecast_data.unit_price) if forecast_data.unit_price else None,
                'created_at': forecast_data.created_at.isoformat() if forecast_data.created_at else None,
                'updated_at': forecast_data.updated_at.isoformat() if forecast_data.updated_at else None
            })
        
        total_pages = (total_records + request.page_size - 1) // request.page_size
        
        return DataViewResponse(
            data=data,
            total_records=total_records,
            page=request.page,
            page_size=request.page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing database data: {str(e)}")

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and store data file in database"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        if 'date' not in df.columns or 'quantity' not in df.columns:
            raise HTTPException(status_code=400, detail="Data must contain 'date' and 'quantity' columns")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        df = df.dropna(subset=['quantity'])
        
        text_columns = ['product', 'product_group', 'product_hierarchy', 
                       'customer', 'customer_group', 'customer_region',
                       'location', 'location_region', 
                       'ship_to_party', 'sold_to_party', 'uom']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('nan', None)
        
        inserted_count = 0
        duplicate_count = 0
        error_count = 0
        
        BATCH_SIZE = 1000
        
        dimension_cache = {'product': {}, 'customer': {}, 'location': {}}
        records = df.to_dict('records')
        
        # Process dimensions
        for row_data in records:
            try:
                if row_data.get('product') and row_data['product'] not in dimension_cache['product']:
                    product_id = DimensionManager.get_or_create_dimension_cached(
                        db, 'product', row_data['product'],
                        product_group=row_data.get('product_group'),
                        product_hierarchy=row_data.get('product_hierarchy')
                    )
                    dimension_cache['product'][row_data['product']] = product_id
                
                if row_data.get('customer') and row_data['customer'] not in dimension_cache['customer']:
                    customer_id = DimensionManager.get_or_create_dimension_cached(
                        db, 'customer', row_data['customer'],
                        customer_group=row_data.get('customer_group'),
                        customer_region=row_data.get('customer_region')
                    )
                    dimension_cache['customer'][row_data['customer']] = customer_id
                
                if row_data.get('location') and row_data['location'] not in dimension_cache['location']:
                    location_id = DimensionManager.get_or_create_dimension_cached(
                        db, 'location', row_data['location'],
                        location_region=row_data.get('location_region')
                    )
                    dimension_cache['location'][row_data['location']] = location_id
            except Exception as e:
                print(f"Error processing dimensions: {e}")
                error_count += 1
        
        db.commit()
        
        # Create forecast records
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            batch_records = []
            
            for row_data in batch:
                try:
                    product_id = dimension_cache['product'].get(row_data.get('product'))
                    customer_id = dimension_cache['customer'].get(row_data.get('customer'))
                    location_id = dimension_cache['location'].get(row_data.get('location'))
                    
                    combination_ids = DimensionManager.get_combination_ids(
                        db, product_id, customer_id, location_id
                    )
                    
                    forecast_data_record = ForecastData(
                        product_id=product_id,
                        customer_id=customer_id,
                        location_id=location_id,
                        product_customer_id=combination_ids.get('product_customer_id'),
                        product_location_id=combination_ids.get('product_location_id'),
                        customer_location_id=combination_ids.get('customer_location_id'),
                        product_customer_location_id=combination_ids.get('product_customer_location_id'),
                        quantity=row_data.get('quantity'),
                        uom=row_data.get('uom'),
                        date=row_data.get('date').date(),
                        unit_price=row_data.get('unit_price'),
                        product=row_data.get('product'),
                        customer=row_data.get('customer'),
                        location=row_data.get('location')
                    )
                    batch_records.append(forecast_data_record)
                except Exception as e:
                    print(f"Error creating record: {e}")
                    error_count += 1
                    continue
            
            if batch_records:
                try:
                    db.bulk_save_objects(batch_records)
                    db.commit()
                    inserted_count += len(batch_records)
                except Exception:
                    db.rollback()
                    for record in batch_records:
                        try:
                            db.add(record)
                            db.commit()
                            inserted_count += 1
                        except:
                            db.rollback()
                            duplicate_count += 1
        
        db.commit()
        total_records = db.query(func.count(ForecastData.id)).scalar()
        
        return {
            "message": "File processed successfully",
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "totalRecords": total_records,
            "filename": file.filename
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")