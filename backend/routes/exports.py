from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, User, DimensionManager
from auth import get_current_user
import pandas as pd
import io
from datetime import datetime

router = APIRouter()

@router.post("/download_forecast_excel")
async def download_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download single forecast data as Excel"""
    try:
        from main import ForecastResult
        
        result = ForecastResult(**request['forecastResult'])
        forecast_by = request.get('forecastBy', '')
        selected_item = request.get('selectedItem', '')

        hist = result.historicData
        fore = result.forecastData

        product_name = ''
        customer_name = ''
        location_name = ''

        if result.combination:
            if 'product_id' in result.combination and result.combination['product_id'] is not None:
                product_name = DimensionManager.get_dimension_name(db, 'product', result.combination['product_id'])
            elif 'product' in result.combination:
                product_name = result.combination['product']

            if 'customer_id' in result.combination and result.combination['customer_id'] is not None:
                customer_name = DimensionManager.get_dimension_name(db, 'customer', result.combination['customer_id'])
            elif 'customer' in result.combination:
                customer_name = result.combination['customer']

            if 'location_id' in result.combination and result.combination['location_id'] is not None:
                location_name = DimensionManager.get_dimension_name(db, 'location', result.combination['location_id'])
            elif 'location' in result.combination:
                location_name = result.combination['location']
        else:
            if forecast_by == 'product':
                product_name = selected_item
            elif forecast_by == 'customer':
                customer_name = selected_item
            elif forecast_by == 'location':
                location_name = selected_item

        hist_rows = []
        fore_rows = []

        for d in hist:
            hist_rows.append({
                "Product": product_name,
                "Customer": customer_name,
                "Location": location_name,
                "Date": d.date,
                "Period": d.period,
                "Quantity": d.quantity,
                "Type": "Historical"
            })

        for d in fore:
            fore_rows.append({
                "Product": product_name,
                "Customer": customer_name,
                "Location": location_name,
                "Date": d.date,
                "Period": d.period,
                "Quantity": d.quantity,
                "Type": "Forecast"
            })

        all_rows = hist_rows + fore_rows
        df = pd.DataFrame(all_rows)

        config_df = pd.DataFrame([{
            "Algorithm": result.selectedAlgorithm,
            "Accuracy": result.accuracy,
            "MAE": result.mae,
            "RMSE": result.rmse,
            "Trend": result.trend,
            "Historic Periods": len(result.historicData),
            "Forecast Periods": len(result.forecastData)
        }])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast Data")
            config_df.to_excel(writer, index=False, sheet_name="Configuration")

            if result.allAlgorithms:
                algo_data = []
                for algo in result.allAlgorithms:
                    algo_data.append({
                        "Algorithm": algo.algorithm,
                        "Accuracy": algo.accuracy,
                        "MAE": algo.mae,
                        "RMSE": algo.rmse,
                        "Trend": algo.trend
                    })
                algo_df = pd.DataFrame(algo_data)
                algo_df.to_excel(writer, index=False, sheet_name="All Algorithms")

        output.seek(0)

        filename_parts = []
        if product_name:
            safe_product = "".join(c for c in str(product_name) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_product:
                filename_parts.append(safe_product)
        if customer_name:
            safe_customer = "".join(c for c in str(customer_name) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_customer:
                filename_parts.append(safe_customer)
        if location_name:
            safe_location = "".join(c for c in str(location_name) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_location:
                filename_parts.append(safe_location)

        filename_base = "_".join(filename_parts) if filename_parts else "forecast"
        filename = f"{filename_base}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")

@router.post("/download_multi_forecast_excel")
async def download_multi_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download multi-forecast data as Excel with all combinations"""
    try:
        from main import MultiForecastResult
        
        multi_result = MultiForecastResult(**request['multiForecastResult'])

        all_data = []
        summary_data = []

        for result in multi_result.results:
            combination = result.combination or {}
            
            product_name = ''
            customer_name = ''
            location_name = ''

            if 'product_id' in combination and combination['product_id'] is not None:
                product_name = DimensionManager.get_dimension_name(db, 'product', combination['product_id'])
            elif 'product' in combination:
                product_name = combination['product']

            if 'customer_id' in combination and combination['customer_id'] is not None:
                customer_name = DimensionManager.get_dimension_name(db, 'customer', combination['customer_id'])
            elif 'customer' in combination:
                customer_name = combination['customer']

            if 'location_id' in combination and combination['location_id'] is not None:
                location_name = DimensionManager.get_dimension_name(db, 'location', combination['location_id'])
            elif 'location' in combination:
                location_name = combination['location']

            summary_data.append({
                "Product": product_name,
                "Customer": customer_name,
                "Location": location_name,
                "Algorithm": result.selectedAlgorithm,
                "Accuracy": result.accuracy,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "Trend": result.trend,
                "Historic_Periods": len(result.historicData),
                "Forecast_Periods": len(result.forecastData)
            })

            for d in result.historicData:
                all_data.append({
                    "Product": product_name,
                    "Customer": customer_name,
                    "Location": location_name,
                    "Date": d.date,
                    "Period": d.period,
                    "Quantity": d.quantity,
                    "Type": "Historical",
                    "Algorithm": result.selectedAlgorithm,
                    "Accuracy": result.accuracy
                })

            for d in result.forecastData:
                all_data.append({
                    "Product": product_name,
                    "Customer": customer_name,
                    "Location": location_name,
                    "Date": d.date,
                    "Period": d.period,
                    "Quantity": d.quantity,
                    "Type": "Forecast",
                    "Algorithm": result.selectedAlgorithm,
                    "Accuracy": result.accuracy
                })

        all_df = pd.DataFrame(all_data)
        summary_df = pd.DataFrame(summary_data)

        best_combo_display = ""
        if multi_result.summary['bestCombination']['combination']:
            best_combo_dict = multi_result.summary['bestCombination']['combination']
            p_name = DimensionManager.get_dimension_name(db, 'product', best_combo_dict.get('product_id')) if best_combo_dict.get('product_id') else best_combo_dict.get('product', '')
            c_name = DimensionManager.get_dimension_name(db, 'customer', best_combo_dict.get('customer_id')) if best_combo_dict.get('customer_id') else best_combo_dict.get('customer', '')
            l_name = DimensionManager.get_dimension_name(db, 'location', best_combo_dict.get('location_id')) if best_combo_dict.get('location_id') else best_combo_dict.get('location', '')
            best_combo_display = f"{p_name} → {c_name} → {l_name}"

        worst_combo_display = ""
        if multi_result.summary['worstCombination']['combination']:
            worst_combo_dict = multi_result.summary['worstCombination']['combination']
            p_name = DimensionManager.get_dimension_name(db, 'product', worst_combo_dict.get('product_id')) if worst_combo_dict.get('product_id') else worst_combo_dict.get('product', '')
            c_name = DimensionManager.get_dimension_name(db, 'customer', worst_combo_dict.get('customer_id')) if worst_combo_dict.get('customer_id') else worst_combo_dict.get('customer', '')
            l_name = DimensionManager.get_dimension_name(db, 'location', worst_combo_dict.get('location_id')) if worst_combo_dict.get('location_id') else worst_combo_dict.get('location', '')
            worst_combo_display = f"{p_name} → {c_name} → {l_name}"

        overall_summary = pd.DataFrame([{
            "Total_Combinations": multi_result.totalCombinations,
            "Successful_Combinations": multi_result.summary['successfulCombinations'],
            "Failed_Combinations": multi_result.summary['failedCombinations'],
            "Average_Accuracy": multi_result.summary['averageAccuracy'],
            "Best_Combination": best_combo_display,
            "Best_Accuracy": multi_result.summary['bestCombination']['accuracy'],
            "Worst_Combination": worst_combo_display,
            "Worst_Accuracy": multi_result.summary['worstCombination']['accuracy']
        }])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            all_df.to_excel(writer, index=False, sheet_name="All Forecast Data")
            summary_df.to_excel(writer, index=False, sheet_name="Combination Summary")
            overall_summary.to_excel(writer, index=False, sheet_name="Overall Summary")

            if multi_result.summary['failedCombinations'] > 0:
                failed_df = pd.DataFrame(multi_result.summary['failedDetails'])
                failed_df.to_excel(writer, index=False, sheet_name="Failed Combinations")

        output.seek(0)

        filename = f"multi_forecast_{multi_result.totalCombinations}combinations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating multi-forecast Excel: {str(e)}")