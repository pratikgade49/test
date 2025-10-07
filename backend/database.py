#!/usr/bin/env python3
"""
Database configuration and models for PostgreSQL integration with surrogate key system

"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, UniqueConstraint, Boolean, DECIMAL, Text # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore
from sqlalchemy.orm import Session
from passlib.context import CryptContext # type: ignore
import os, hashlib
from enum import Enum as PyEnum
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Make sure which credantals are used to connect locally or in cloud environment
DB_HOST = os.getenv("DB_HOST", "localhost") #172.17.0.1
DB_PORT = os.getenv("DB_PORT", "5433")  #5432
DB_USER = os.getenv("DB_USER", "postgres")  # Changed from root to postgres
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "forecasting_db")

# Database configuration
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Create engine with better error handling
try:
    engine = create_engine(DATABASE_URL, echo=False)
except Exception as e:
    print(f"❌ Error connecting to database: {e}")
    print("Please ensure:")
    print("1. PostgreSQL server is running")
    print("2. Database 'forecasting_db' exists")
    print("3. Connection credentials are correct")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dimension Tables for Surrogate Keys
class ProductDimension(Base):
    """Dimension table for products with surrogate keys"""
    __tablename__ = "product_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), unique=True, nullable=False, index=True)
    product_group = Column(String(255), nullable=True)
    product_hierarchy = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CustomerDimension(Base):
    """Dimension table for customers with surrogate keys"""
    __tablename__ = "customer_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(255), unique=True, nullable=False, index=True)
    customer_group = Column(String(255), nullable=True)
    customer_region = Column(String(255), nullable=True)
    ship_to_party = Column(String(255), nullable=True)
    sold_to_party = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LocationDimension(Base):
    """Dimension table for locations with surrogate keys"""
    __tablename__ = "location_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    location_name = Column(String(255), unique=True, nullable=False, index=True)
    location_region = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 2D Combination Tables
class ProductCustomerCombination(Base):
    """2D combination table for product-customer pairs"""
    __tablename__ = "product_customer_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'customer_id', name='unique_product_customer'),
    )

class ProductLocationCombination(Base):
    """2D combination table for product-location pairs"""
    __tablename__ = "product_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'location_id', name='unique_product_location'),
    )

class CustomerLocationCombination(Base):
    """2D combination table for customer-location pairs"""
    __tablename__ = "customer_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('customer_id', 'location_id', name='unique_customer_location'),
    )

# 3D Combination Table
class ProductCustomerLocationCombination(Base):
    """3D combination table for product-customer-location triplets"""
    __tablename__ = "product_customer_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'customer_id', 'location_id', name='unique_product_customer_location'),
    )

# Helper functions for surrogate key management
def generate_combination_key(*args):
    """Generate a stable hash key for combinations"""
    combined = "_".join(str(arg) for arg in sorted(args))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def get_or_create_dimension(db: Session, model_class, name_field: str, name_value: str, **extra_fields):
    """Get or create a dimension record and return its ID"""
    # Try to find existing record
    existing = db.query(model_class).filter(getattr(model_class, name_field) == name_value).first()
    if existing:
        return existing.id
    
    # Create new record
    new_record = model_class(**{name_field: name_value, **extra_fields})
    db.add(new_record)
    db.flush()  # Get the ID without committing
    return new_record.id

def get_or_create_combination(db: Session, model_class, combination_key: str, **key_fields):
    """Get or create a combination record and return its ID"""
    # Try to find existing record
    existing = db.query(model_class).filter(model_class.combination_key == combination_key).first()
    if existing:
        return existing.id
    
    # Create new record
    new_record = model_class(combination_key=combination_key, **key_fields)
    db.add(new_record)
    db.flush()  # Get the ID without committing
    return new_record.id

# Original models with modifications

class ScheduleFrequency(PyEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ScheduleStatus(PyEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ScheduledForecast(Base):
    """Model for storing scheduled forecast configurations"""
    __tablename__ = "scheduled_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)  # JSON string of ForecastConfig
    frequency = Column(String(20), nullable=False)  # Use String instead of Enum for SQLAlchemy compatibility
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)  # Optional end date
    next_run = Column(DateTime, nullable=False)
    last_run = Column(DateTime, nullable=True)
    status = Column(String(20), default='active')  # Use String instead of Enum
    run_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ForecastExecution(Base):
    """Model for storing forecast execution history"""
    __tablename__ = "forecast_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    scheduled_forecast_id = Column(Integer, nullable=False, index=True)
    execution_time = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)  # success, failed, running
    duration_seconds = Column(Integer, nullable=True)
    result_summary = Column(Text, nullable=True)  # JSON summary of results
    error_message = Column(Text, nullable=True)
    forecast_data = Column(Text, nullable=True)  # JSON of forecast results
    created_at = Column(DateTime, default=datetime.utcnow)

class ExternalFactorData(Base):
    __tablename__ = 'external_factor_data'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    factor_name = Column(String(255), index=True)
    factor_value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('date', 'factor_name', name='unique_external_factor_record'),
    )

class User(Base):
    """Model for user authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_approved = Column(Boolean, default=False)  # New: requires admin approval
    is_admin = Column(Boolean, default=False)      # New: admin flag
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
class ForecastData(Base):
    """Model for storing forecast data from Excel uploads"""
    __tablename__ = "forecast_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Surrogate key references to dimension tables
    product_id = Column(Integer, nullable=True, index=True)
    customer_id = Column(Integer, nullable=True, index=True)
    location_id = Column(Integer, nullable=True, index=True)
    
    # Combination keys for faster filtering
    product_customer_id = Column(Integer, nullable=True, index=True)
    product_location_id = Column(Integer, nullable=True, index=True)
    customer_location_id = Column(Integer, nullable=True, index=True)
    product_customer_location_id = Column(Integer, nullable=True, index=True)
    
    # Core data fields
    quantity = Column(DECIMAL(15, 2), nullable=False)
    uom = Column(String(50), nullable=True)
    date = Column(Date, nullable=False)
    unit_price = Column(DECIMAL(15, 2), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Legacy string columns for backward compatibility during migration
    product = Column(String(255), nullable=True)
    customer = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    
    # Unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint(
            'product_id', 'customer_id', 'location_id', 'date',
            name='unique_forecast_record'
        ),
    )

class SavedForecastResult(Base):
    """Model for storing user's saved forecast results"""
    __tablename__ = "saved_forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)  # Foreign key to users table
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)  # JSON string of ForecastConfig
    forecast_data = Column(Text, nullable=False)  # JSON string of ForecastResult
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for user + name combination
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_user_forecast_name'),
    )

class ForecastConfiguration(Base):
    """Model for storing forecast configurations"""
    __tablename__ = "forecast_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Store both keys and names for flexibility
    forecast_by = Column(String(50), nullable=False)  # 'product', 'customer', 'location'
    selected_item_id = Column(Integer, nullable=True)  # Surrogate key for single selection
    selected_item_name = Column(String(255), nullable=True)  # Human readable name
    
    # For advanced mode (exact combinations)
    selected_product_id = Column(Integer, nullable=True)
    selected_customer_id = Column(Integer, nullable=True)
    selected_location_id = Column(Integer, nullable=True)
    
    # Multi-select support (JSON arrays of IDs)
    selected_items_ids = Column(Text, nullable=True)  # JSON array of surrogate keys
    algorithm = Column(String(100), nullable=False, default='best_fit')
    interval = Column(String(20), nullable=False, default='month')
    historic_period = Column(Integer, nullable=False, default=12)
    forecast_period = Column(Integer, nullable=False, default=6)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for configuration name
    __table_args__ = (
        UniqueConstraint('name', name='unique_config_name'),
    )

# Import model persistence tables
from model_persistence import SavedModel, ModelAccuracyHistory
# Data access layer functions for surrogate key operations
class DimensionManager:
    """Manager class for handling dimension and combination operations"""
    
    # Class level caches
    _dimension_cache = {
        'product': {},
        'customer': {},
        'location': {}
    }
    
    _combination_cache = {
        'product_customer': {},
        'product_location': {},
        'customer_location': {},
        'product_customer_location': {}
    }
    
    @staticmethod
    def get_or_create_dimension_cached(db: Session, dimension_type: str, name: str, **extra_fields) -> int:
        """Get or create dimension with caching"""
        if not name:
            return None
            
        # Check cache first
        if name in DimensionManager._dimension_cache[dimension_type]:
            return DimensionManager._dimension_cache[dimension_type][name]
        
        # Not in cache, query database
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(ProductDimension.product_name == name).first()
            if not record:
                record = ProductDimension(product_name=name, **extra_fields)
                db.add(record)
                db.flush()
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(CustomerDimension.customer_name == name).first()
            if not record:
                record = CustomerDimension(customer_name=name, **extra_fields)
                db.add(record)
                db.flush()
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(LocationDimension.location_name == name).first()
            if not record:
                record = LocationDimension(location_name=name, **extra_fields)
                db.add(record)
                db.flush()
        
        # Cache the result
        DimensionManager._dimension_cache[dimension_type][name] = record.id
        return record.id
    
    @staticmethod
    def get_combination_ids(db: Session, product_id: int = None, customer_id: int = None, location_id: int = None) -> dict:
        """Get all combination IDs efficiently using caching"""
        result = {}
        
        # Generate combination keys
        if product_id and customer_id:
            combo_key = f"{product_id}_{customer_id}"
            if combo_key in DimensionManager._combination_cache['product_customer']:
                result['product_customer_id'] = DimensionManager._combination_cache['product_customer'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, customer_id)
                combo_id = get_or_create_combination(
                    db, ProductCustomerCombination, combo_key_hash,
                    product_id=product_id, customer_id=customer_id
                )
                DimensionManager._combination_cache['product_customer'][combo_key] = combo_id
                result['product_customer_id'] = combo_id
        
        if product_id and location_id:
            combo_key = f"{product_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['product_location']:
                result['product_location_id'] = DimensionManager._combination_cache['product_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, location_id)
                combo_id = get_or_create_combination(
                    db, ProductLocationCombination, combo_key_hash,
                    product_id=product_id, location_id=location_id
                )
                DimensionManager._combination_cache['product_location'][combo_key] = combo_id
                result['product_location_id'] = combo_id
        
        if customer_id and location_id:
            combo_key = f"{customer_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['customer_location']:
                result['customer_location_id'] = DimensionManager._combination_cache['customer_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(customer_id, location_id)
                combo_id = get_or_create_combination(
                    db, CustomerLocationCombination, combo_key_hash,
                    customer_id=customer_id, location_id=location_id
                )
                DimensionManager._combination_cache['customer_location'][combo_key] = combo_id
                result['customer_location_id'] = combo_id
        
        if product_id and customer_id and location_id:
            combo_key = f"{product_id}_{customer_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['product_customer_location']:
                result['product_customer_location_id'] = DimensionManager._combination_cache['product_customer_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, customer_id, location_id)
                combo_id = get_or_create_combination(
                    db, ProductCustomerLocationCombination, combo_key_hash,
                    product_id=product_id, customer_id=customer_id, location_id=location_id
                )
                DimensionManager._combination_cache['product_customer_location'][combo_key] = combo_id
                result['product_customer_location_id'] = combo_id
        
        return result

    @staticmethod
    def process_forecast_row(db: Session, row_data: dict) -> dict:
        """Process a single forecast data row and return surrogate keys"""
        result = {}
        
        # Extract dimension values and ensure they're strings
        product_name = str(row_data.get('product')) if row_data.get('product') is not None else None
        customer_name = str(row_data.get('customer')) if row_data.get('customer') is not None else None
        location_name = str(row_data.get('location')) if row_data.get('location') is not None else None
        
        # Get or create dimension IDs using cached methods
        if product_name:
            result['product_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'product', product_name,
                product_group=row_data.get('product_group'),
                product_hierarchy=row_data.get('product_hierarchy')
            )
        
        if customer_name:
            result['customer_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'customer', customer_name,
                customer_group=row_data.get('customer_group'),
                customer_region=row_data.get('customer_region'),
                ship_to_party=row_data.get('ship_to_party'),
                sold_to_party=row_data.get('sold_to_party')
            )
        
        if location_name:
            result['location_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'location', location_name,
                location_region=row_data.get('location_region')
            )
        
        # Get combination IDs using cached method
        if result.get('product_id') or result.get('customer_id') or result.get('location_id'):
            combination_ids = DimensionManager.get_combination_ids(
                db,
                product_id=result.get('product_id'),
                customer_id=result.get('customer_id'),
                location_id=result.get('location_id')
            )
            result.update(combination_ids)
        
        return result
    
    @staticmethod
    def get_dimension_name(db: Session, dimension_type: str, dimension_id: int) -> str:
        """Get human-readable name from dimension ID"""
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(ProductDimension.id == dimension_id).first()
            return record.product_name if record else f"Product_{dimension_id}"
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(CustomerDimension.id == dimension_id).first()
            return record.customer_name if record else f"Customer_{dimension_id}"
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(LocationDimension.id == dimension_id).first()
            return record.location_name if record else f"Location_{dimension_id}"
        return f"Unknown_{dimension_id}"
    
    @staticmethod
    def get_dimension_id(db: Session, dimension_type: str, dimension_name: str) -> int:
        """Get dimension ID from human-readable name"""
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(ProductDimension.product_name == dimension_name).first()
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(CustomerDimension.customer_name == dimension_name).first()
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(LocationDimension.location_name == dimension_name).first()
        else:
            return None
        return record.id if record else None

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables (assumes database already exists)"""
    try:
        # Test connection first
        with engine.connect() as connection:
            print("✅ Database connection successful!")
        
        # Create tables if they don't exist
        return create_tables()
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("Please ensure:")
        print("1. PostgreSQL server is running")
        print("2. Database 'forecasting_db' exists")
        print("3. User has proper permissions")
        return False
    print("3. Connection credentials are correct")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ExternalFactorData(Base):
    __tablename__ = 'external_factor_data'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    factor_name = Column(String(255), index=True)
    factor_value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('date', 'factor_name', name='unique_external_factor_record'),
    )

class User(Base):
    """Model for user authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_approved = Column(Boolean, default=False)  # New: requires admin approval
    is_admin = Column(Boolean, default=False)      # New: admin flag
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)



def create_default_user():
    """Create default admin user if no users exist"""
    try:
        db = SessionLocal()
        user_count = db.query(User).count()
        if user_count == 0:
            default_user = User(
                username="admin",
                email="admin@forecasting.com",
                hashed_password=User.hash_password("admin123"),
                full_name="System Administrator",
                is_active=True,
                is_approved=True,
                is_admin=True
            )
            db.add(default_user)
            db.commit()
            print("✅ Default admin user created (username: admin, password: admin123)")
        db.close()
    except Exception as e:
        print(f"⚠️  Error creating default user: {e}")

# Import model persistence tables
from model_persistence import SavedModel, ModelAccuracyHistory

def create_tables():
    """Create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        # Also create model persistence tables
        SavedModel.metadata.create_all(bind=engine)
        ModelAccuracyHistory.metadata.create_all(bind=engine)
        print("✅ Database tables verified/created successfully!")
        print("✅ Saved forecast results table created!")
        create_default_user()
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables (assumes database already exists)"""
    try:
        # Test connection first
        with engine.connect() as connection:
            print("✅ Database connection successful!")
        
        # Create tables if they don't exist
        return create_tables()
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("Please ensure:")
        print("1. PostgreSQL server is running")
        print("2. Database 'forecasting_db' exists")
        print("3. User has proper permissions")
        return False