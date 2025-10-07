from fastapi import APIRouter
from .auth import router as auth_router
from .forecasting import router as forecasting_router
from .data_management import router as data_management_router
from .configurations import router as configurations_router
from .scheduled_forecasts import router as scheduled_forecasts_router
from .external_factors import router as external_factors_router
from .admin import router as admin_router
from .saved_forecasts import router as saved_forecasts_router
from .exports import router as exports_router
from .model_cache import router as model_cache_router

# Combine all routers
api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(forecasting_router, tags=["Forecasting"])
api_router.include_router(data_management_router, prefix="/database", tags=["Data Management"])
api_router.include_router(configurations_router, prefix="/configurations", tags=["Configurations"])
api_router.include_router(scheduled_forecasts_router, prefix="/scheduled_forecasts", tags=["Scheduled Forecasts"])
api_router.include_router(external_factors_router, tags=["External Factors"])
api_router.include_router(admin_router, prefix="/admin", tags=["Admin"])
api_router.include_router(saved_forecasts_router, prefix="/saved_forecasts", tags=["Saved Forecasts"])
api_router.include_router(exports_router, tags=["Exports"])
api_router.include_router(model_cache_router, tags=["Model Cache"])