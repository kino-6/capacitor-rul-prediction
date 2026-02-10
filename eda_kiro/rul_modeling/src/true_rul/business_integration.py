"""
Business Systems Integration

This module provides integration capabilities with enterprise business systems including:
- ERP system integration for maintenance workflows
- Financial system integration for cost tracking
- Supply chain integration for parts procurement
- Asset management system integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
import sqlite3
import csv
from enum import Enum

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Integration status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    SYNCING = "syncing"

@dataclass
class WorkOrder:
    """Work order data structure for ERP integration"""
    work_order_id: str
    equipment_id: str
    maintenance_type: str
    priority: int
    scheduled_date: str
    estimated_duration_hours: float
    estimated_cost: float
    status: str  # created, scheduled, in_progress, completed, cancelled
    assigned_technician: Optional[str] = None
    parts_required: Optional[List[Dict[str, Any]]] = None
    actual_duration_hours: Optional[float] = None
    actual_cost: Optional[float] = None
    completion_date: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FinancialTransaction:
    """Financial transaction for cost tracking"""
    transaction_id: str
    work_order_id: str
    equipment_id: str
    transaction_type: str  # maintenance_cost, parts_cost, labor_cost, downtime_cost
    amount: float
    currency: str
    transaction_date: str
    cost_center: str
    account_code: str
    description: str
    approved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PartsProcurement:
    """Parts procurement request for supply chain integration"""
    procurement_id: str
    work_order_id: str
    equipment_id: str
    part_number: str
    part_description: str
    quantity_required: int
    unit_cost: float
    supplier: str
    requested_date: str
    required_date: str
    status: str  # requested, approved, ordered, received, cancelled
    lead_time_days: int
    inventory_available: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AssetRecord:
    """Asset record for asset management integration"""
    asset_id: str
    equipment_id: str
    asset_tag: str
    description: str
    location: str
    department: str
    acquisition_date: str
    acquisition_cost: float
    current_value: float
    depreciation_method: str
    useful_life_years: int
    maintenance_history: List[Dict[str, Any]]
    warranty_expiry: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BusinessSystemConnector(ABC):
    """Abstract base class for business system connectors"""
    
    def __init__(self, system_name: str, connection_config: Dict[str, Any]):
        self.system_name = system_name
        self.connection_config = connection_config
        self.status = IntegrationStatus.DISCONNECTED
        self.last_sync = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the business system"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the business system"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the business system"""
        pass
    
    @abstractmethod
    def sync_data(self) -> Dict[str, Any]:
        """Synchronize data with the business system"""
        pass

class ERPConnector(BusinessSystemConnector):
    """ERP system connector for maintenance workflow integration"""
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__("ERP", connection_config)
        self.api_base_url = connection_config.get('api_base_url', '')
        self.api_key = connection_config.get('api_key', '')
        self.username = connection_config.get('username', '')
        self.password = connection_config.get('password', '')
        self.timeout = connection_config.get('timeout', 30)
        
    def connect(self) -> bool:
        """Connect to ERP system"""
        try:
            # Simulate ERP connection
            if self.api_base_url and (self.api_key or (self.username and self.password)):
                self.status = IntegrationStatus.CONNECTED
                logger.info(f"Connected to ERP system: {self.api_base_url}")
                return True
            else:
                logger.error("Invalid ERP connection configuration")
                self.status = IntegrationStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to ERP system: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from ERP system"""
        try:
            self.status = IntegrationStatus.DISCONNECTED
            logger.info("Disconnected from ERP system")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from ERP system: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test ERP connection"""
        try:
            if self.status == IntegrationStatus.CONNECTED:
                # Simulate connection test
                return True
            return False
        except Exception as e:
            logger.error(f"ERP connection test failed: {e}")
            return False
    
    def sync_data(self) -> Dict[str, Any]:
        """Sync data with ERP system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to ERP system'}
            
            self.status = IntegrationStatus.SYNCING
            
            # Simulate data sync
            sync_result = {
                'success': True,
                'work_orders_synced': 25,
                'equipment_records_updated': 50,
                'sync_timestamp': datetime.now().isoformat()
            }
            
            self.status = IntegrationStatus.CONNECTED
            self.last_sync = datetime.now()
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Error syncing with ERP system: {e}")
            self.status = IntegrationStatus.ERROR
            return {'success': False, 'error': str(e)}
    
    def create_work_order(self, work_order: WorkOrder) -> Dict[str, Any]:
        """Create work order in ERP system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to ERP system'}
            
            # Simulate work order creation
            logger.info(f"Creating work order {work_order.work_order_id} in ERP system")
            
            # In real implementation, this would make API call to ERP
            # For simulation, we'll just return success
            return {
                'success': True,
                'work_order_id': work_order.work_order_id,
                'erp_reference': f"WO-{work_order.work_order_id}-{datetime.now().strftime('%Y%m%d')}",
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating work order in ERP: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_work_order_status(self, work_order_id: str, status: str, 
                                completion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update work order status in ERP system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to ERP system'}
            
            logger.info(f"Updating work order {work_order_id} status to {status} in ERP system")
            
            # Simulate status update
            return {
                'success': True,
                'work_order_id': work_order_id,
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating work order status in ERP: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_equipment_master_data(self, equipment_ids: List[str]) -> Dict[str, Any]:
        """Get equipment master data from ERP system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to ERP system'}
            
            # Simulate equipment data retrieval
            equipment_data = []
            for eq_id in equipment_ids:
                equipment_data.append({
                    'equipment_id': eq_id,
                    'description': f'Equipment {eq_id}',
                    'location': f'Zone_{np.random.randint(1, 6)}',
                    'cost_center': f'CC_{np.random.randint(1000, 9999)}',
                    'maintenance_plan': f'MP_{eq_id}',
                    'criticality': np.random.choice(['low', 'medium', 'high'])
                })
            
            return {
                'success': True,
                'equipment_data': equipment_data,
                'retrieved_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving equipment data from ERP: {e}")
            return {'success': False, 'error': str(e)}

class FinancialSystemConnector(BusinessSystemConnector):
    """Financial system connector for cost tracking integration"""
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__("Financial", connection_config)
        self.database_path = connection_config.get('database_path', 'financial_system.db')
        self.chart_of_accounts = connection_config.get('chart_of_accounts', {})
        
    def connect(self) -> bool:
        """Connect to financial system"""
        try:
            # Initialize SQLite database for simulation
            self.conn = sqlite3.connect(self.database_path)
            self._initialize_financial_tables()
            self.status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to financial system: {self.database_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to financial system: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from financial system"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            self.status = IntegrationStatus.DISCONNECTED
            logger.info("Disconnected from financial system")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from financial system: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test financial system connection"""
        try:
            if self.status == IntegrationStatus.CONNECTED and hasattr(self, 'conn'):
                # Test database connection
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                return True
            return False
        except Exception as e:
            logger.error(f"Financial system connection test failed: {e}")
            return False
    
    def sync_data(self) -> Dict[str, Any]:
        """Sync data with financial system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to financial system'}
            
            self.status = IntegrationStatus.SYNCING
            
            # Simulate financial data sync
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM financial_transactions")
            transaction_count = cursor.fetchone()[0]
            
            sync_result = {
                'success': True,
                'transactions_synced': transaction_count,
                'cost_centers_updated': 15,
                'sync_timestamp': datetime.now().isoformat()
            }
            
            self.status = IntegrationStatus.CONNECTED
            self.last_sync = datetime.now()
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Error syncing with financial system: {e}")
            self.status = IntegrationStatus.ERROR
            return {'success': False, 'error': str(e)}
    
    def record_maintenance_cost(self, transaction: FinancialTransaction) -> Dict[str, Any]:
        """Record maintenance cost in financial system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to financial system'}
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO financial_transactions 
                (transaction_id, work_order_id, equipment_id, transaction_type, 
                 amount, currency, transaction_date, cost_center, account_code, 
                 description, approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.transaction_id, transaction.work_order_id, transaction.equipment_id,
                transaction.transaction_type, transaction.amount, transaction.currency,
                transaction.transaction_date, transaction.cost_center, transaction.account_code,
                transaction.description, transaction.approved
            ))
            
            self.conn.commit()
            
            logger.info(f"Recorded maintenance cost transaction {transaction.transaction_id}")
            
            return {
                'success': True,
                'transaction_id': transaction.transaction_id,
                'recorded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error recording maintenance cost: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_cost_center_budget(self, cost_center: str, period: str) -> Dict[str, Any]:
        """Get budget information for cost center"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to financial system'}
            
            # Simulate budget retrieval
            budget_data = {
                'cost_center': cost_center,
                'period': period,
                'total_budget': np.random.uniform(50000, 200000),
                'maintenance_budget': np.random.uniform(10000, 50000),
                'spent_to_date': np.random.uniform(5000, 30000),
                'remaining_budget': 0,  # Will be calculated
                'budget_utilization': 0  # Will be calculated
            }
            
            budget_data['remaining_budget'] = budget_data['maintenance_budget'] - budget_data['spent_to_date']
            budget_data['budget_utilization'] = budget_data['spent_to_date'] / budget_data['maintenance_budget']
            
            return {
                'success': True,
                'budget_data': budget_data,
                'retrieved_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving budget data: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_cost_report(self, start_date: str, end_date: str, 
                           cost_centers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate cost report for specified period"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to financial system'}
            
            cursor = self.conn.cursor()
            
            # Build query based on parameters
            query = """
                SELECT cost_center, transaction_type, SUM(amount) as total_amount, COUNT(*) as transaction_count
                FROM financial_transactions 
                WHERE transaction_date BETWEEN ? AND ?
            """
            params = [start_date, end_date]
            
            if cost_centers:
                placeholders = ','.join(['?' for _ in cost_centers])
                query += f" AND cost_center IN ({placeholders})"
                params.extend(cost_centers)
            
            query += " GROUP BY cost_center, transaction_type ORDER BY cost_center, transaction_type"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Process results
            cost_report = {}
            for row in results:
                cost_center, transaction_type, total_amount, transaction_count = row
                if cost_center not in cost_report:
                    cost_report[cost_center] = {}
                
                cost_report[cost_center][transaction_type] = {
                    'total_amount': total_amount,
                    'transaction_count': transaction_count
                }
            
            return {
                'success': True,
                'cost_report': cost_report,
                'period': f"{start_date} to {end_date}",
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return {'success': False, 'error': str(e)}
    
    def _initialize_financial_tables(self):
        """Initialize financial system database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_transactions (
                transaction_id TEXT PRIMARY KEY,
                work_order_id TEXT,
                equipment_id TEXT,
                transaction_type TEXT,
                amount REAL,
                currency TEXT,
                transaction_date TEXT,
                cost_center TEXT,
                account_code TEXT,
                description TEXT,
                approved BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_centers (
                cost_center TEXT PRIMARY KEY,
                description TEXT,
                manager TEXT,
                budget_amount REAL,
                period TEXT
            )
        """)
        
        self.conn.commit()

class SupplyChainConnector(BusinessSystemConnector):
    """Supply chain connector for parts procurement integration"""
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__("SupplyChain", connection_config)
        self.inventory_database = connection_config.get('inventory_database', 'inventory.db')
        self.supplier_catalog = connection_config.get('supplier_catalog', {})
        
    def connect(self) -> bool:
        """Connect to supply chain system"""
        try:
            # Initialize inventory database
            self.conn = sqlite3.connect(self.inventory_database)
            self._initialize_inventory_tables()
            self.status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to supply chain system: {self.inventory_database}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to supply chain system: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from supply chain system"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            self.status = IntegrationStatus.DISCONNECTED
            logger.info("Disconnected from supply chain system")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from supply chain system: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test supply chain connection"""
        try:
            if self.status == IntegrationStatus.CONNECTED and hasattr(self, 'conn'):
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                return True
            return False
        except Exception as e:
            logger.error(f"Supply chain connection test failed: {e}")
            return False
    
    def sync_data(self) -> Dict[str, Any]:
        """Sync data with supply chain system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to supply chain system'}
            
            self.status = IntegrationStatus.SYNCING
            
            # Simulate inventory sync
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inventory_items")
            item_count = cursor.fetchone()[0]
            
            sync_result = {
                'success': True,
                'inventory_items_synced': item_count,
                'procurement_requests_processed': 12,
                'supplier_updates': 5,
                'sync_timestamp': datetime.now().isoformat()
            }
            
            self.status = IntegrationStatus.CONNECTED
            self.last_sync = datetime.now()
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Error syncing with supply chain system: {e}")
            self.status = IntegrationStatus.ERROR
            return {'success': False, 'error': str(e)}
    
    def create_procurement_request(self, procurement: PartsProcurement) -> Dict[str, Any]:
        """Create parts procurement request"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to supply chain system'}
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO procurement_requests 
                (procurement_id, work_order_id, equipment_id, part_number, part_description,
                 quantity_required, unit_cost, supplier, requested_date, required_date,
                 status, lead_time_days, inventory_available)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                procurement.procurement_id, procurement.work_order_id, procurement.equipment_id,
                procurement.part_number, procurement.part_description, procurement.quantity_required,
                procurement.unit_cost, procurement.supplier, procurement.requested_date,
                procurement.required_date, procurement.status, procurement.lead_time_days,
                procurement.inventory_available
            ))
            
            self.conn.commit()
            
            logger.info(f"Created procurement request {procurement.procurement_id}")
            
            return {
                'success': True,
                'procurement_id': procurement.procurement_id,
                'estimated_delivery': (datetime.now() + timedelta(days=procurement.lead_time_days)).isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating procurement request: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_inventory_availability(self, part_numbers: List[str]) -> Dict[str, Any]:
        """Check inventory availability for parts"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to supply chain system'}
            
            cursor = self.conn.cursor()
            
            availability_data = {}
            for part_number in part_numbers:
                cursor.execute("""
                    SELECT part_number, description, quantity_on_hand, quantity_reserved,
                           unit_cost, supplier, lead_time_days
                    FROM inventory_items 
                    WHERE part_number = ?
                """, (part_number,))
                
                result = cursor.fetchone()
                if result:
                    part_num, description, on_hand, reserved, unit_cost, supplier, lead_time = result
                    available = on_hand - reserved
                    
                    availability_data[part_number] = {
                        'description': description,
                        'quantity_available': available,
                        'quantity_on_hand': on_hand,
                        'quantity_reserved': reserved,
                        'unit_cost': unit_cost,
                        'supplier': supplier,
                        'lead_time_days': lead_time,
                        'in_stock': available > 0
                    }
                else:
                    availability_data[part_number] = {
                        'description': 'Unknown part',
                        'quantity_available': 0,
                        'in_stock': False,
                        'requires_procurement': True
                    }
            
            return {
                'success': True,
                'availability_data': availability_data,
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking inventory availability: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_supplier_information(self, part_number: str) -> Dict[str, Any]:
        """Get supplier information for a part"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to supply chain system'}
            
            # Simulate supplier lookup
            suppliers = [
                {
                    'supplier_name': 'Industrial Parts Co.',
                    'contact_email': 'orders@industrialparts.com',
                    'lead_time_days': np.random.randint(5, 15),
                    'unit_cost': np.random.uniform(50, 500),
                    'minimum_order_quantity': np.random.randint(1, 10),
                    'reliability_rating': np.random.uniform(0.8, 0.98)
                },
                {
                    'supplier_name': 'Equipment Solutions Ltd.',
                    'contact_email': 'procurement@equipmentsolutions.com',
                    'lead_time_days': np.random.randint(7, 20),
                    'unit_cost': np.random.uniform(45, 480),
                    'minimum_order_quantity': np.random.randint(1, 5),
                    'reliability_rating': np.random.uniform(0.85, 0.95)
                }
            ]
            
            return {
                'success': True,
                'part_number': part_number,
                'suppliers': suppliers,
                'retrieved_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving supplier information: {e}")
            return {'success': False, 'error': str(e)}
    
    def _initialize_inventory_tables(self):
        """Initialize inventory database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_items (
                part_number TEXT PRIMARY KEY,
                description TEXT,
                quantity_on_hand INTEGER,
                quantity_reserved INTEGER,
                unit_cost REAL,
                supplier TEXT,
                lead_time_days INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procurement_requests (
                procurement_id TEXT PRIMARY KEY,
                work_order_id TEXT,
                equipment_id TEXT,
                part_number TEXT,
                part_description TEXT,
                quantity_required INTEGER,
                unit_cost REAL,
                supplier TEXT,
                requested_date TEXT,
                required_date TEXT,
                status TEXT,
                lead_time_days INTEGER,
                inventory_available INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert sample inventory data
        sample_parts = [
            ('CAP-001', 'Electrolytic Capacitor 1000uF', 25, 5, 45.50, 'Industrial Parts Co.', 7),
            ('MOT-001', 'AC Motor Bearing Set', 12, 2, 125.00, 'Equipment Solutions Ltd.', 10),
            ('PMP-001', 'Pump Impeller Assembly', 8, 1, 275.00, 'Industrial Parts Co.', 14),
            ('CMP-001', 'Compressor Valve Kit', 15, 3, 185.00, 'Equipment Solutions Ltd.', 12)
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO inventory_items 
            (part_number, description, quantity_on_hand, quantity_reserved, unit_cost, supplier, lead_time_days)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_parts)
        
        self.conn.commit()

class AssetManagementConnector(BusinessSystemConnector):
    """Asset management system connector"""
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__("AssetManagement", connection_config)
        self.asset_database = connection_config.get('asset_database', 'assets.db')
        
    def connect(self) -> bool:
        """Connect to asset management system"""
        try:
            self.conn = sqlite3.connect(self.asset_database)
            self._initialize_asset_tables()
            self.status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to asset management system: {self.asset_database}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to asset management system: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from asset management system"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            self.status = IntegrationStatus.DISCONNECTED
            logger.info("Disconnected from asset management system")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from asset management system: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test asset management connection"""
        try:
            if self.status == IntegrationStatus.CONNECTED and hasattr(self, 'conn'):
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                return True
            return False
        except Exception as e:
            logger.error(f"Asset management connection test failed: {e}")
            return False
    
    def sync_data(self) -> Dict[str, Any]:
        """Sync data with asset management system"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to asset management system'}
            
            self.status = IntegrationStatus.SYNCING
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assets")
            asset_count = cursor.fetchone()[0]
            
            sync_result = {
                'success': True,
                'assets_synced': asset_count,
                'maintenance_records_updated': 45,
                'depreciation_calculated': asset_count,
                'sync_timestamp': datetime.now().isoformat()
            }
            
            self.status = IntegrationStatus.CONNECTED
            self.last_sync = datetime.now()
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Error syncing with asset management system: {e}")
            self.status = IntegrationStatus.ERROR
            return {'success': False, 'error': str(e)}
    
    def get_asset_information(self, equipment_id: str) -> Dict[str, Any]:
        """Get comprehensive asset information"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to asset management system'}
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT asset_id, equipment_id, asset_tag, description, location, department,
                       acquisition_date, acquisition_cost, current_value, depreciation_method,
                       useful_life_years, warranty_expiry
                FROM assets 
                WHERE equipment_id = ?
            """, (equipment_id,))
            
            result = cursor.fetchone()
            if result:
                asset_data = {
                    'asset_id': result[0],
                    'equipment_id': result[1],
                    'asset_tag': result[2],
                    'description': result[3],
                    'location': result[4],
                    'department': result[5],
                    'acquisition_date': result[6],
                    'acquisition_cost': result[7],
                    'current_value': result[8],
                    'depreciation_method': result[9],
                    'useful_life_years': result[10],
                    'warranty_expiry': result[11]
                }
                
                # Get maintenance history
                cursor.execute("""
                    SELECT maintenance_date, maintenance_type, cost, description
                    FROM maintenance_history 
                    WHERE equipment_id = ?
                    ORDER BY maintenance_date DESC
                    LIMIT 10
                """, (equipment_id,))
                
                maintenance_history = []
                for row in cursor.fetchall():
                    maintenance_history.append({
                        'maintenance_date': row[0],
                        'maintenance_type': row[1],
                        'cost': row[2],
                        'description': row[3]
                    })
                
                asset_data['maintenance_history'] = maintenance_history
                
                return {
                    'success': True,
                    'asset_data': asset_data,
                    'retrieved_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'Asset not found for equipment {equipment_id}'
                }
                
        except Exception as e:
            logger.error(f"Error retrieving asset information: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_asset_value(self, asset_id: str, new_value: float, 
                          depreciation_date: str) -> Dict[str, Any]:
        """Update asset current value"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to asset management system'}
            
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE assets 
                SET current_value = ?, last_depreciation_date = ?
                WHERE asset_id = ?
            """, (new_value, depreciation_date, asset_id))
            
            self.conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Updated asset {asset_id} value to ${new_value}")
                return {
                    'success': True,
                    'asset_id': asset_id,
                    'new_value': new_value,
                    'updated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'Asset {asset_id} not found'
                }
                
        except Exception as e:
            logger.error(f"Error updating asset value: {e}")
            return {'success': False, 'error': str(e)}
    
    def record_maintenance_history(self, equipment_id: str, maintenance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record maintenance activity in asset history"""
        try:
            if self.status != IntegrationStatus.CONNECTED:
                return {'success': False, 'error': 'Not connected to asset management system'}
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO maintenance_history 
                (equipment_id, maintenance_date, maintenance_type, cost, description, work_order_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                equipment_id,
                maintenance_data.get('maintenance_date', datetime.now().isoformat()),
                maintenance_data.get('maintenance_type', 'unknown'),
                maintenance_data.get('cost', 0.0),
                maintenance_data.get('description', ''),
                maintenance_data.get('work_order_id', '')
            ))
            
            self.conn.commit()
            
            logger.info(f"Recorded maintenance history for equipment {equipment_id}")
            
            return {
                'success': True,
                'equipment_id': equipment_id,
                'recorded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error recording maintenance history: {e}")
            return {'success': False, 'error': str(e)}
    
    def _initialize_asset_tables(self):
        """Initialize asset management database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                equipment_id TEXT UNIQUE,
                asset_tag TEXT,
                description TEXT,
                location TEXT,
                department TEXT,
                acquisition_date TEXT,
                acquisition_cost REAL,
                current_value REAL,
                depreciation_method TEXT,
                useful_life_years INTEGER,
                warranty_expiry TEXT,
                last_depreciation_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT,
                maintenance_date TEXT,
                maintenance_type TEXT,
                cost REAL,
                description TEXT,
                work_order_id TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert sample asset data
        sample_assets = [
            ('AST-001', 'EQ_001', 'TAG-001', 'Industrial Capacitor Unit', 'Zone_1', 'Production', 
             '2020-01-15', 15000.0, 12000.0, 'straight_line', 10, '2025-01-15'),
            ('AST-002', 'EQ_002', 'TAG-002', 'AC Motor Assembly', 'Zone_2', 'Production',
             '2019-06-20', 25000.0, 18000.0, 'straight_line', 15, '2024-06-20'),
            ('AST-003', 'EQ_003', 'TAG-003', 'Centrifugal Pump', 'Zone_3', 'Utilities',
             '2021-03-10', 18000.0, 15500.0, 'straight_line', 12, '2026-03-10')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO assets 
            (asset_id, equipment_id, asset_tag, description, location, department,
             acquisition_date, acquisition_cost, current_value, depreciation_method,
             useful_life_years, warranty_expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_assets)
        
        self.conn.commit()

class BusinessIntegrationManager:
    """Manager for all business system integrations"""
    
    def __init__(self, integration_config: Dict[str, Dict[str, Any]]):
        """
        Initialize business integration manager
        
        Args:
            integration_config: Configuration for all business system integrations
        """
        self.integration_config = integration_config
        self.connectors = {}
        self.integration_status = {}
        
        # Initialize connectors
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize all business system connectors"""
        try:
            if 'erp' in self.integration_config:
                self.connectors['erp'] = ERPConnector(self.integration_config['erp'])
            
            if 'financial' in self.integration_config:
                self.connectors['financial'] = FinancialSystemConnector(self.integration_config['financial'])
            
            if 'supply_chain' in self.integration_config:
                self.connectors['supply_chain'] = SupplyChainConnector(self.integration_config['supply_chain'])
            
            if 'asset_management' in self.integration_config:
                self.connectors['asset_management'] = AssetManagementConnector(self.integration_config['asset_management'])
            
            logger.info(f"Initialized {len(self.connectors)} business system connectors")
            
        except Exception as e:
            logger.error(f"Error initializing connectors: {e}")
    
    def connect_all_systems(self) -> Dict[str, bool]:
        """Connect to all configured business systems"""
        connection_results = {}
        
        for system_name, connector in self.connectors.items():
            try:
                success = connector.connect()
                connection_results[system_name] = success
                self.integration_status[system_name] = connector.status
                
                if success:
                    logger.info(f"Successfully connected to {system_name}")
                else:
                    logger.error(f"Failed to connect to {system_name}")
                    
            except Exception as e:
                logger.error(f"Error connecting to {system_name}: {e}")
                connection_results[system_name] = False
                self.integration_status[system_name] = IntegrationStatus.ERROR
        
        return connection_results
    
    def disconnect_all_systems(self) -> Dict[str, bool]:
        """Disconnect from all business systems"""
        disconnection_results = {}
        
        for system_name, connector in self.connectors.items():
            try:
                success = connector.disconnect()
                disconnection_results[system_name] = success
                self.integration_status[system_name] = connector.status
                
            except Exception as e:
                logger.error(f"Error disconnecting from {system_name}: {e}")
                disconnection_results[system_name] = False
        
        return disconnection_results
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        status_report = {}
        
        for system_name, connector in self.connectors.items():
            status_report[system_name] = {
                'status': connector.status.value,
                'last_sync': connector.last_sync.isoformat() if connector.last_sync else None,
                'connection_test': connector.test_connection()
            }
        
        return status_report
    
    def sync_all_systems(self) -> Dict[str, Dict[str, Any]]:
        """Synchronize data with all connected systems"""
        sync_results = {}
        
        for system_name, connector in self.connectors.items():
            if connector.status == IntegrationStatus.CONNECTED:
                try:
                    result = connector.sync_data()
                    sync_results[system_name] = result
                    
                except Exception as e:
                    logger.error(f"Error syncing {system_name}: {e}")
                    sync_results[system_name] = {'success': False, 'error': str(e)}
            else:
                sync_results[system_name] = {'success': False, 'error': 'System not connected'}
        
        return sync_results
    
    def create_integrated_work_order(self, maintenance_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create work order across integrated systems"""
        try:
            results = {}
            
            # Create work order in ERP
            if 'erp' in self.connectors:
                work_order = WorkOrder(
                    work_order_id=f"WO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    equipment_id=maintenance_schedule['equipment_id'],
                    maintenance_type=maintenance_schedule['maintenance_type'],
                    priority=maintenance_schedule['priority'],
                    scheduled_date=maintenance_schedule['scheduled_date'],
                    estimated_duration_hours=maintenance_schedule['estimated_duration_hours'],
                    estimated_cost=maintenance_schedule['estimated_cost'],
                    status='created'
                )
                
                erp_result = self.connectors['erp'].create_work_order(work_order)
                results['erp'] = erp_result
                
                # Record financial transaction
                if 'financial' in self.connectors and erp_result.get('success'):
                    transaction = FinancialTransaction(
                        transaction_id=f"TXN_{work_order.work_order_id}_{datetime.now().strftime('%H%M%S')}",
                        work_order_id=work_order.work_order_id,
                        equipment_id=work_order.equipment_id,
                        transaction_type='maintenance_cost',
                        amount=work_order.estimated_cost,
                        currency='USD',
                        transaction_date=datetime.now().isoformat(),
                        cost_center='MAINT_001',
                        account_code='6200',
                        description=f'{work_order.maintenance_type} maintenance for {work_order.equipment_id}',
                        approved=False
                    )
                    
                    financial_result = self.connectors['financial'].record_maintenance_cost(transaction)
                    results['financial'] = financial_result
                
                # Create procurement requests if parts needed
                if 'supply_chain' in self.connectors and maintenance_schedule.get('parts_required'):
                    procurement_results = []
                    for part in maintenance_schedule['parts_required']:
                        procurement = PartsProcurement(
                            procurement_id=f"PR_{work_order.work_order_id}_{part['part_number']}_{datetime.now().strftime('%H%M%S')}",
                            work_order_id=work_order.work_order_id,
                            equipment_id=work_order.equipment_id,
                            part_number=part['part_number'],
                            part_description=part['description'],
                            quantity_required=part['quantity'],
                            unit_cost=part['unit_cost'],
                            supplier=part.get('supplier', 'TBD'),
                            requested_date=datetime.now().isoformat(),
                            required_date=work_order.scheduled_date,
                            status='requested',
                            lead_time_days=part.get('lead_time_days', 7),
                            inventory_available=0
                        )
                        
                        proc_result = self.connectors['supply_chain'].create_procurement_request(procurement)
                        procurement_results.append(proc_result)
                    
                    results['supply_chain'] = procurement_results
            
            return {
                'success': True,
                'work_order_id': work_order.work_order_id if 'work_order' in locals() else None,
                'integration_results': results,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating integrated work order: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_integration_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive integration status report"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"business_integration_report_{timestamp}.json"
            
            # Collect integration data
            integration_status = self.get_integration_status()
            sync_results = {}
            
            # Get recent sync results for connected systems
            for system_name, connector in self.connectors.items():
                if connector.status == IntegrationStatus.CONNECTED:
                    sync_results[system_name] = connector.sync_data()
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'integration_status': integration_status,
                'sync_results': sync_results,
                'configured_systems': list(self.connectors.keys()),
                'connected_systems': [name for name, status in integration_status.items() 
                                    if status['status'] == 'connected'],
                'summary': {
                    'total_systems': len(self.connectors),
                    'connected_systems': len([s for s in integration_status.values() 
                                            if s['status'] == 'connected']),
                    'integration_health': 'healthy' if all(s['connection_test'] for s in integration_status.values()) else 'degraded'
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Business integration report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating integration report: {e}")
            raise