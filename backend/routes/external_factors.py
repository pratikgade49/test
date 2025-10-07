from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db, User, ExternalFactorData
from auth import get_current_user
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy import distinct, func
from sqlalchemy.exc import IntegrityError
import pandas as pd
import io
import requests
from datetime import datetime

router = APIRouter()

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = "82a8e6191d71f41b22cf33bf73f7a0c2"

class FredDataRequest(BaseModel):
    series_ids: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FredDataResponse(BaseModel):
    message: str
    inserted: int
    duplicates: int
    series_processed: int
    series_details: List[Dict[str, Any]]

@router.get("/external_factors")
async def get_external_factors(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique values for external factors from database"""
    try:
        factors = db.query(distinct(ExternalFactorData.factor_name)).filter(
            ExternalFactorData.factor_name.isnot(None)
        ).all()

        return {
            "external_factors": sorted([f[0] for f in factors if f[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting external factors: {str(e)}")

@router.post("/upload_external_factors")
async def upload_external_factors(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and store external factor data file"""
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")

        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        if 'date' not in df.columns or 'factor_name' not in df.columns or 'factor_value' not in df.columns:
            raise HTTPException(status_code=400, detail="Data must contain 'date', 'factor_name', and 'factor_value' columns")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        df['factor_value'] = pd.to_numeric(df['factor_value'], errors='coerce')
        df = df.dropna(subset=['factor_value'])

        inserted_count = 0
        duplicate_count = 0

        for _, row in df.iterrows():
            date_value = row['date']
            if hasattr(date_value, 'date'):
                date_value = date_value.date()
            
            record_data = {
                'date': date_value,
                'factor_name': row['factor_name'],
                'factor_value': row['factor_value']
            }
            
            try:
                new_record = ExternalFactorData(**record_data)
                db.add(new_record)
                db.flush()
                inserted_count += 1
            except IntegrityError:
                db.rollback()
                duplicate_count += 1
            except Exception as e:
                db.rollback()
                print(f"Error adding record: {e}")
                continue

        db.commit()

        total_records = db.query(func.count(ExternalFactorData.id)).scalar()

        return {
            "message": "File processed and stored in database successfully",
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "totalRecords": total_records,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/fetch_fred_data", response_model=FredDataResponse)
async def fetch_fred_data(
    request: FredDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch live economic data from FRED API and store in database"""
    if not FRED_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="FRED API key not configured. Please set FRED_API_KEY environment variable."
        )

    cleaned_api_key = FRED_API_KEY.strip().lstrip('+')

    try:
        total_inserted = 0
        total_duplicates = 0
        series_details = []

        for series_id in request.series_ids:
            try:
                if not series_id or not isinstance(series_id, str):
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'Invalid series ID provided',
                        'inserted': 0
                    })
                    continue

                params = {
                    'series_id': series_id.upper().strip(),
                    'api_key': cleaned_api_key,
                    'file_type': 'json',
                    'limit': 1000
                }

                if request.start_date:
                    if isinstance(request.start_date, str):
                        params['observation_start'] = request.start_date
                    else:
                        params['observation_start'] = request.start_date.strftime('%Y-%m-%d')

                if request.end_date:
                    if isinstance(request.end_date, str):
                        params['observation_end'] = request.end_date
                    else:
                        params['observation_end'] = request.end_date.strftime('%Y-%m-%d')

                print(f"Making FRED API request for series: {series_id}")

                response = requests.get(
                    FRED_API_BASE_URL, 
                    params=params, 
                    timeout=30,
                    headers={'User-Agent': 'YourApp/1.0'}
                )

                response.raise_for_status()
                data = response.json()

                if 'error_message' in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': f'FRED API error: {data["error_message"]}',
                        'inserted': 0
                    })
                    continue

                if 'observations' not in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'No observations found in API response',
                        'inserted': 0
                    })
                    continue

                observations = data['observations']

                if not observations:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'warning',
                        'message': 'No data available for the specified date range',
                        'inserted': 0
                    })
                    continue

                records_to_insert = []
                existing_records = set()

                existing_query = db.query(ExternalFactorData.date, ExternalFactorData.factor_name).all()
                for rec in existing_query:
                    existing_records.add((rec.date, rec.factor_name))

                inserted_count = 0
                duplicate_count = 0
                skipped_count = 0

                for obs in observations:
                    try:
                        obs_date = pd.to_datetime(obs['date']).date()

                        if obs['value'] == '.' or obs['value'] is None or obs['value'] == '':
                            skipped_count += 1
                            continue

                        obs_value = float(obs['value'])

                        if (obs_date, series_id) not in existing_records:
                            record_data = ExternalFactorData(
                                date=obs_date,
                                factor_name=series_id,
                                factor_value=obs_value
                            )
                            records_to_insert.append(record_data)
                            inserted_count += 1
                        else:
                            duplicate_count += 1

                    except (ValueError, TypeError) as e:
                        print(f"Error processing observation for {series_id}: {e}")
                        skipped_count += 1
                        continue

                if records_to_insert:
                    try:
                        db.bulk_save_objects(records_to_insert)
                        db.commit()
                        print(f"Successfully inserted {len(records_to_insert)} records for {series_id}")
                    except Exception as db_error:
                        db.rollback()
                        series_details.append({
                            'series_id': series_id,
                            'status': 'error',
                            'message': f'Database insertion failed: {str(db_error)}',
                            'inserted': 0
                        })
                        continue

                total_inserted += inserted_count
                total_duplicates += duplicate_count

                message_parts = [f'Successfully processed {len(observations)} observations']
                if skipped_count > 0:
                    message_parts.append(f'{skipped_count} skipped (missing values)')

                series_details.append({
                    'series_id': series_id,
                    'status': 'success',
                    'message': ', '.join(message_parts),
                    'inserted': inserted_count,
                    'duplicates': duplicate_count
                })

            except requests.RequestException as e:
                error_msg = f'API request failed: {str(e)}'
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if 'error_message' in error_data:
                            error_msg += f' - {error_data["error_message"]}'
                    except:
                        error_msg += f' - HTTP {e.response.status_code}'

                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': error_msg,
                    'inserted': 0
                })
            except Exception as e:
                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': f'Processing failed: {str(e)}',
                    'inserted': 0
                })

        return FredDataResponse(
            message=f"FRED data fetch completed. Processed {len(request.series_ids)} series.",
            inserted=total_inserted,
            duplicates=total_duplicates,
            series_processed=len(request.series_ids),
            series_details=series_details
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching FRED data: {str(e)}")

@router.get("/fred_series_info")
async def get_fred_series_info(
    current_user: User = Depends(get_current_user)
):
    """Get information about popular FRED series for users"""
    popular_series = {
        "Economic Indicators": {
            "GDP": "Gross Domestic Product",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "PAYEMS": "All Employees, Total Nonfarm",
            "INDPRO": "Industrial Production Index"
        },
        "Financial Markets": {
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS3MO": "3-Month Treasury Constant Maturity Rate",
            "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
            "DEXJPUS": "Japan / U.S. Foreign Exchange Rate"
        },
        "Business & Trade": {
            "HOUST": "Housing Starts",
            "RSAFS": "Advance Retail Sales",
            "IMPGS": "Imports of Goods and Services",
            "EXPGS": "Exports of Goods and Services"
        }
    }

    return {
        "message": "Popular FRED series for economic forecasting",
        "series": popular_series,
        "note": "Visit https://fred.stlouisfed.org to explore more series"
    }