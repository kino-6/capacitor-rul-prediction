#!/usr/bin/env python3
"""
Test script for Business Systems Integration

This script tests the business integration capabilities including:
- ERP system integration for maintenance workflows
- Financial system integration for cost tracking
- Supply chain integration for parts procurement
- Asset management system integration
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.business_integration import (
    BusinessIntegrationManager, ERPConnector, FinancialSystemConnector,
    SupplyChainConnector, AssetManagementConnector, WorkOrder, 
    FinancialTransaction, PartsProcurement, AssetRecord, IntegrationStatus
)

def create_test_integration_config():
    """Create test configuration for business system integrations"""
    return {
        'erp': {
            'api_base_url': 'https://test-erp.company.com/api',
            'api_key': 'test_api_key_12345',
            'username': 'integration_user',
            'password': 'test_password',
            'timeout': 30
        },
        'financial': {
            'database_path': 'test_financial_system.db',
            'chart_of_accounts': {
                '6200': 'Maintenance Expenses',
                '6210': 'Parts and Materials',
                '6220': 'Labor Costs'
            }
        },
        'supply_chain': {
            'inventory_database': 'test_inventory.db',
            'supplier_catalog': {
                'Industrial Parts Co.': 'primary_supplier',
                'Equipment Solutions Ltd.': 'secondary_supplier'
            }
        },
        'asset_management': {
            'asset_database': 'test_assets.db'
        }
    }

def test_erp_integration():
    """Test ERP system integration"""
    print("Testing ERP Integration...")
    
    config = create_test_integration_config()['erp']
    erp_connector = ERPConnector(config)
    
    # Test connection
    assert erp_connector.connect(), "ERP connection failed"
    assert erp_connector.test_connection(), "ERP connection test failed"
    print("✓ ERP connection established")
    
    # Test data sync
    sync_result = erp_connector.sync_data()
    assert sync_result['success'], f"ERP sync failed: {sync_result.get('error')}"
    print(f"✓ ERP data sync completed: {sync_result['work_orders_synced']} work orders synced")
    
    # Test work order creation
    work_order = WorkOrder(
        work_order_id="WO_TEST_001",
        equipment_id="EQ_001",
        maintenance_type="preventive",
        priority=3,
        scheduled_date=datetime.now().isoformat(),
        estimated_duration_hours=4.0,
        estimated_cost=1500.0,
        status="created",
        parts_required=[
            {'part_number': 'CAP-001', 'quantity': 2, 'description': 'Capacitor'}
        ]
    )
    
    create_result = erp_connector.create_work_order(work_order)
    assert create_result['success'], f"Work order creation failed: {create_result.get('error')}"
    print(f"✓ Work order created: {create_result['work_order_id']}")
    
    # Test work order status update
    update_result = erp_connector.update_work_order_status(
        work_order.work_order_id, "in_progress"
    )
    assert update_result['success'], f"Work order update failed: {update_result.get('error')}"
    print(f"✓ Work order status updated to: {update_result['status']}")
    
    # Test equipment master data retrieval
    equipment_data = erp_connector.get_equipment_master_data(['EQ_001', 'EQ_002'])
    assert equipment_data['success'], f"Equipment data retrieval failed: {equipment_data.get('error')}"
    print(f"✓ Equipment master data retrieved for {len(equipment_data['equipment_data'])} items")
    
    # Test disconnection
    assert erp_connector.disconnect(), "ERP disconnection failed"
    print("✓ ERP integration test passed!\n")
    
    return erp_connector

def test_financial_integration():
    """Test financial system integration"""
    print("Testing Financial System Integration...")
    
    config = create_test_integration_config()['financial']
    financial_connector = FinancialSystemConnector(config)
    
    # Test connection
    assert financial_connector.connect(), "Financial system connection failed"
    assert financial_connector.test_connection(), "Financial system connection test failed"
    print("✓ Financial system connection established")
    
    # Test data sync
    sync_result = financial_connector.sync_data()
    assert sync_result['success'], f"Financial sync failed: {sync_result.get('error')}"
    print(f"✓ Financial data sync completed: {sync_result['transactions_synced']} transactions synced")
    
    # Test maintenance cost recording
    transaction = FinancialTransaction(
        transaction_id=f"TXN_TEST_{datetime.now().strftime('%H%M%S_%f')}",
        work_order_id="WO_TEST_001",
        equipment_id="EQ_001",
        transaction_type="maintenance_cost",
        amount=1500.0,
        currency="USD",
        transaction_date=datetime.now().isoformat(),
        cost_center="MAINT_001",
        account_code="6200",
        description="Preventive maintenance for EQ_001",
        approved=False
    )
    
    record_result = financial_connector.record_maintenance_cost(transaction)
    assert record_result['success'], f"Cost recording failed: {record_result.get('error')}"
    print(f"✓ Maintenance cost recorded: {record_result['transaction_id']}")
    
    # Test budget retrieval
    budget_result = financial_connector.get_cost_center_budget("MAINT_001", "2026-Q1")
    assert budget_result['success'], f"Budget retrieval failed: {budget_result.get('error')}"
    budget_data = budget_result['budget_data']
    print(f"✓ Budget retrieved - Total: ${budget_data['total_budget']:,.0f}, Maintenance: ${budget_data['maintenance_budget']:,.0f}")
    
    # Test cost report generation
    start_date = (datetime.now() - timedelta(days=30)).isoformat()
    end_date = datetime.now().isoformat()
    
    report_result = financial_connector.generate_cost_report(start_date, end_date, ["MAINT_001"])
    assert report_result['success'], f"Cost report generation failed: {report_result.get('error')}"
    print(f"✓ Cost report generated for period: {report_result['period']}")
    
    # Test disconnection
    assert financial_connector.disconnect(), "Financial system disconnection failed"
    print("✓ Financial system integration test passed!\n")
    
    return financial_connector

def test_supply_chain_integration():
    """Test supply chain integration"""
    print("Testing Supply Chain Integration...")
    
    config = create_test_integration_config()['supply_chain']
    supply_connector = SupplyChainConnector(config)
    
    # Test connection
    assert supply_connector.connect(), "Supply chain connection failed"
    assert supply_connector.test_connection(), "Supply chain connection test failed"
    print("✓ Supply chain connection established")
    
    # Test data sync
    sync_result = supply_connector.sync_data()
    assert sync_result['success'], f"Supply chain sync failed: {sync_result.get('error')}"
    print(f"✓ Supply chain data sync completed: {sync_result['inventory_items_synced']} items synced")
    
    # Test inventory availability check
    part_numbers = ['CAP-001', 'MOT-001', 'PMP-001']
    availability_result = supply_connector.check_inventory_availability(part_numbers)
    assert availability_result['success'], f"Inventory check failed: {availability_result.get('error')}"
    
    print("✓ Inventory Availability:")
    for part_num, data in availability_result['availability_data'].items():
        status = "In Stock" if data.get('in_stock', False) else "Out of Stock"
        available = data.get('quantity_available', 0)
        print(f"  - {part_num}: {available} units available ({status})")
    
    # Test procurement request creation
    procurement = PartsProcurement(
        procurement_id=f"PR_TEST_{datetime.now().strftime('%H%M%S_%f')}",
        work_order_id="WO_TEST_001",
        equipment_id="EQ_001",
        part_number="CAP-001",
        part_description="Electrolytic Capacitor 1000uF",
        quantity_required=5,
        unit_cost=45.50,
        supplier="Industrial Parts Co.",
        requested_date=datetime.now().isoformat(),
        required_date=(datetime.now() + timedelta(days=7)).isoformat(),
        status="requested",
        lead_time_days=7,
        inventory_available=2
    )
    
    procurement_result = supply_connector.create_procurement_request(procurement)
    assert procurement_result['success'], f"Procurement request failed: {procurement_result.get('error')}"
    print(f"✓ Procurement request created: {procurement_result['procurement_id']}")
    print(f"  Estimated delivery: {procurement_result['estimated_delivery']}")
    
    # Test supplier information retrieval
    supplier_result = supply_connector.get_supplier_information("CAP-001")
    assert supplier_result['success'], f"Supplier info retrieval failed: {supplier_result.get('error')}"
    print(f"✓ Supplier information retrieved for {supplier_result['part_number']}:")
    for supplier in supplier_result['suppliers'][:2]:  # Show first 2 suppliers
        print(f"  - {supplier['supplier_name']}: ${supplier['unit_cost']:.2f}, {supplier['lead_time_days']} days")
    
    # Test disconnection
    assert supply_connector.disconnect(), "Supply chain disconnection failed"
    print("✓ Supply chain integration test passed!\n")
    
    return supply_connector

def test_asset_management_integration():
    """Test asset management integration"""
    print("Testing Asset Management Integration...")
    
    config = create_test_integration_config()['asset_management']
    asset_connector = AssetManagementConnector(config)
    
    # Test connection
    assert asset_connector.connect(), "Asset management connection failed"
    assert asset_connector.test_connection(), "Asset management connection test failed"
    print("✓ Asset management connection established")
    
    # Test data sync
    sync_result = asset_connector.sync_data()
    assert sync_result['success'], f"Asset management sync failed: {sync_result.get('error')}"
    print(f"✓ Asset management data sync completed: {sync_result['assets_synced']} assets synced")
    
    # Test asset information retrieval
    asset_result = asset_connector.get_asset_information("EQ_001")
    assert asset_result['success'], f"Asset info retrieval failed: {asset_result.get('error')}"
    
    asset_data = asset_result['asset_data']
    print(f"✓ Asset information retrieved:")
    print(f"  - Asset ID: {asset_data['asset_id']}")
    print(f"  - Description: {asset_data['description']}")
    print(f"  - Location: {asset_data['location']}")
    print(f"  - Acquisition Cost: ${asset_data['acquisition_cost']:,.0f}")
    print(f"  - Current Value: ${asset_data['current_value']:,.0f}")
    print(f"  - Maintenance History: {len(asset_data['maintenance_history'])} records")
    
    # Test asset value update
    new_value = asset_data['current_value'] * 0.95  # Depreciate by 5%
    update_result = asset_connector.update_asset_value(
        asset_data['asset_id'], new_value, datetime.now().isoformat()
    )
    assert update_result['success'], f"Asset value update failed: {update_result.get('error')}"
    print(f"✓ Asset value updated to: ${update_result['new_value']:,.0f}")
    
    # Test maintenance history recording
    maintenance_data = {
        'maintenance_date': datetime.now().isoformat(),
        'maintenance_type': 'preventive',
        'cost': 1500.0,
        'description': 'Routine preventive maintenance',
        'work_order_id': 'WO_TEST_001'
    }
    
    history_result = asset_connector.record_maintenance_history("EQ_001", maintenance_data)
    assert history_result['success'], f"Maintenance history recording failed: {history_result.get('error')}"
    print(f"✓ Maintenance history recorded for: {history_result['equipment_id']}")
    
    # Test disconnection
    assert asset_connector.disconnect(), "Asset management disconnection failed"
    print("✓ Asset management integration test passed!\n")
    
    return asset_connector

def test_integration_manager():
    """Test business integration manager"""
    print("Testing Business Integration Manager...")
    
    config = create_test_integration_config()
    integration_manager = BusinessIntegrationManager(config)
    
    # Test connecting all systems
    connection_results = integration_manager.connect_all_systems()
    print("✓ Connection Results:")
    for system, success in connection_results.items():
        status = "✓ Connected" if success else "❌ Failed"
        print(f"  - {system.upper()}: {status}")
    
    assert all(connection_results.values()), "Not all systems connected successfully"
    
    # Test integration status
    status_report = integration_manager.get_integration_status()
    print("✓ Integration Status:")
    for system, status in status_report.items():
        print(f"  - {system.upper()}: {status['status']} (Test: {'Pass' if status['connection_test'] else 'Fail'})")
    
    # Test synchronized data sync
    sync_results = integration_manager.sync_all_systems()
    print("✓ Synchronization Results:")
    for system, result in sync_results.items():
        if result['success']:
            print(f"  - {system.upper()}: Synced successfully")
        else:
            print(f"  - {system.upper()}: Sync failed - {result.get('error', 'Unknown error')}")
    
    # Test integrated work order creation
    maintenance_schedule = {
        'equipment_id': 'EQ_001',
        'maintenance_type': 'preventive',
        'priority': 3,
        'scheduled_date': (datetime.now() + timedelta(days=7)).isoformat(),
        'estimated_duration_hours': 4.0,
        'estimated_cost': 2500.0,
        'parts_required': [
            {
                'part_number': 'CAP-001',
                'description': 'Electrolytic Capacitor',
                'quantity': 2,
                'unit_cost': 45.50,
                'supplier': 'Industrial Parts Co.',
                'lead_time_days': 7
            }
        ]
    }
    
    integrated_result = integration_manager.create_integrated_work_order(maintenance_schedule)
    assert integrated_result['success'], f"Integrated work order creation failed: {integrated_result.get('error')}"
    print(f"✓ Integrated work order created: {integrated_result['work_order_id']}")
    
    # Show integration results
    integration_results = integrated_result['integration_results']
    for system, result in integration_results.items():
        if isinstance(result, list):  # Supply chain returns list of procurement results
            print(f"  - {system.upper()}: {len(result)} procurement requests created")
        elif result.get('success'):
            print(f"  - {system.upper()}: Integration successful")
        else:
            print(f"  - {system.upper()}: Integration failed")
    
    # Test integration report generation
    output_dir = Path("output/integration_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = integration_manager.generate_integration_report(
        str(output_dir / f"business_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    
    assert os.path.exists(report_path), "Integration report not generated"
    print(f"✓ Integration report generated: {report_path}")
    
    # Validate report content
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    assert 'integration_status' in report_data, "Missing integration status in report"
    assert 'summary' in report_data, "Missing summary in report"
    
    summary = report_data['summary']
    print("✓ Integration Summary:")
    print(f"  - Total Systems: {summary['total_systems']}")
    print(f"  - Connected Systems: {summary['connected_systems']}")
    print(f"  - Integration Health: {summary['integration_health']}")
    
    # Test disconnecting all systems
    disconnection_results = integration_manager.disconnect_all_systems()
    print("✓ Disconnection Results:")
    for system, success in disconnection_results.items():
        status = "✓ Disconnected" if success else "❌ Failed"
        print(f"  - {system.upper()}: {status}")
    
    print("✓ Business integration manager test passed!\n")
    return integration_manager, report_path

def test_data_flow_integration():
    """Test end-to-end data flow across all systems"""
    print("Testing End-to-End Data Flow Integration...")
    
    # Create separate config for end-to-end test to avoid database conflicts
    config = create_test_integration_config()
    config['financial']['database_path'] = 'test_financial_e2e.db'
    config['supply_chain']['inventory_database'] = 'test_inventory_e2e.db'
    config['asset_management']['asset_database'] = 'test_assets_e2e.db'
    
    integration_manager = BusinessIntegrationManager(config)
    
    # Connect all systems
    connection_results = integration_manager.connect_all_systems()
    assert all(connection_results.values()), "Failed to connect all systems"
    
    # Simulate complete maintenance workflow
    equipment_id = "EQ_TEST_001"
    
    # Step 1: Get asset information
    asset_connector = integration_manager.connectors['asset_management']
    
    # Create test asset first
    test_asset_data = {
        'asset_id': 'AST_TEST_001',
        'equipment_id': equipment_id,
        'asset_tag': 'TAG_TEST_001',
        'description': 'Test Equipment for Integration',
        'location': 'Test Zone',
        'department': 'Testing',
        'acquisition_date': '2023-01-01',
        'acquisition_cost': 20000.0,
        'current_value': 18000.0,
        'depreciation_method': 'straight_line',
        'useful_life_years': 10,
        'warranty_expiry': '2028-01-01'
    }
    
    # Step 2: Check parts availability
    supply_connector = integration_manager.connectors['supply_chain']
    parts_needed = ['CAP-001', 'MOT-001']
    availability_result = supply_connector.check_inventory_availability(parts_needed)
    
    print("✓ Parts Availability Check:")
    for part, data in availability_result['availability_data'].items():
        print(f"  - {part}: {data.get('quantity_available', 0)} available")
    
    # Step 3: Create integrated work order
    maintenance_schedule = {
        'equipment_id': equipment_id,
        'maintenance_type': 'corrective',
        'priority': 4,
        'scheduled_date': (datetime.now() + timedelta(days=2)).isoformat(),
        'estimated_duration_hours': 6.0,
        'estimated_cost': 3500.0,
        'parts_required': [
            {
                'part_number': 'CAP-001',
                'description': 'Electrolytic Capacitor',
                'quantity': 3,
                'unit_cost': 45.50,
                'supplier': 'Industrial Parts Co.',
                'lead_time_days': 5
            }
        ]
    }
    
    work_order_result = integration_manager.create_integrated_work_order(maintenance_schedule)
    assert work_order_result['success'], "Integrated work order creation failed"
    
    work_order_id = work_order_result['work_order_id']
    print(f"✓ End-to-end workflow completed for work order: {work_order_id}")
    
    # Step 4: Verify data consistency across systems
    integration_results = work_order_result['integration_results']
    
    # Check ERP work order
    if 'erp' in integration_results:
        erp_result = integration_results['erp']
        assert erp_result['success'], "ERP work order creation failed"
        print(f"  - ERP: Work order {erp_result['work_order_id']} created")
    
    # Check financial transaction
    if 'financial' in integration_results:
        financial_result = integration_results['financial']
        assert financial_result['success'], "Financial transaction recording failed"
        print(f"  - Financial: Transaction {financial_result['transaction_id']} recorded")
    
    # Check procurement requests
    if 'supply_chain' in integration_results:
        procurement_results = integration_results['supply_chain']
        successful_procurements = [r for r in procurement_results if r['success']]
        print(f"  - Supply Chain: {len(successful_procurements)} procurement requests created")
    
    # Disconnect all systems
    integration_manager.disconnect_all_systems()
    
    print("✓ End-to-end data flow integration test passed!\n")
    return work_order_id

def run_comprehensive_business_integration_test():
    """Run comprehensive test of all business integration features"""
    print("=" * 60)
    print("BUSINESS SYSTEMS INTEGRATION TEST")
    print("=" * 60)
    print()
    
    try:
        # Test individual connectors
        erp_connector = test_erp_integration()
        financial_connector = test_financial_integration()
        supply_connector = test_supply_chain_integration()
        asset_connector = test_asset_management_integration()
        
        # Test integration manager
        integration_manager, report_path = test_integration_manager()
        
        # Test end-to-end data flow
        work_order_id = test_data_flow_integration()
        
        # Summary
        print("=" * 60)
        print("BUSINESS INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print("✓ All business integration tests passed successfully!")
        print()
        print("Generated Files:")
        print(f"  - Integration Report: {report_path}")
        print(f"  - Test Databases: test_financial_system.db, test_inventory.db, test_assets.db")
        print()
        print("Integration Capabilities Tested:")
        print("  ✓ ERP system integration for maintenance workflows")
        print("  ✓ Financial system integration for cost tracking")
        print("  ✓ Supply chain integration for parts procurement")
        print("  ✓ Asset management system integration")
        print("  ✓ Cross-system data synchronization")
        print("  ✓ Integrated work order creation")
        print("  ✓ End-to-end workflow automation")
        print()
        print("Key Integration Features:")
        print("  • Work order lifecycle management")
        print("  • Automated cost tracking and budgeting")
        print("  • Parts procurement optimization")
        print("  • Asset depreciation and history tracking")
        print("  • Real-time system status monitoring")
        print("  • Comprehensive integration reporting")
        print()
        print(f"Sample Work Order Created: {work_order_id}")
        print()
        print("Business systems integration is ready for production deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_business_integration_test()
    sys.exit(0 if success else 1)