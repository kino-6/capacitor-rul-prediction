#!/usr/bin/env python3
"""
Test script for the regulatory compliance module
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.regulatory_compliance import (
    ComplianceManager,
    ValidationProtocolManager,
    ValidationExecutor,
    AuditTrailManager,
    ComplianceReportGenerator,
    ComplianceStandard,
    ValidationProtocol,
    ValidationResult,
    AuditTrailEntry,
    ComplianceReport
)


def test_validation_protocol_manager():
    """Test validation protocol manager"""
    print("Testing ValidationProtocolManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            manager = ValidationProtocolManager(protocols_dir=temp_dir)
            
            # Test ISO 13485 protocol creation
            iso_protocol = manager.create_iso_13485_protocol()
            assert iso_protocol.standard == ComplianceStandard.ISO_13485
            assert len(iso_protocol.requirements) > 0
            assert len(iso_protocol.test_procedures) > 0
            print("  ✓ ISO 13485 protocol created successfully")
            
            # Test FDA 21 CFR protocol creation
            fda_protocol = manager.create_fda_21cfr_protocol()
            assert fda_protocol.standard == ComplianceStandard.FDA_21CFR
            assert len(fda_protocol.requirements) > 0
            print("  ✓ FDA 21 CFR protocol created successfully")
            
            # Test ISO 9001 protocol creation
            iso9001_protocol = manager.create_iso_9001_protocol()
            assert iso9001_protocol.standard == ComplianceStandard.ISO_9001
            assert len(iso9001_protocol.requirements) > 0
            print("  ✓ ISO 9001 protocol created successfully")
            
            # Test protocol retrieval
            retrieved_protocol = manager.get_protocol(iso_protocol.protocol_id)
            assert retrieved_protocol is not None
            assert retrieved_protocol.protocol_id == iso_protocol.protocol_id
            print("  ✓ Protocol retrieval working")
            
            # Test protocol listing
            protocols = manager.list_protocols()
            assert len(protocols) >= 3
            print("  ✓ Protocol listing working")
            
            print("✓ ValidationProtocolManager tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error testing ValidationProtocolManager: {e}")
            return False


def test_audit_trail_manager():
    """Test audit trail manager"""
    print("Testing AuditTrailManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            manager = AuditTrailManager(audit_dir=temp_dir)
            
            # Test prediction logging
            from true_rul.data_structures import PredictionResult
            
            pred_result = PredictionResult(
                rul_cycles=50,
                rul_confidence_lower=45,
                rul_confidence_upper=55,
                degradation_score=0.3,
                degradation_stage="early_degradation",
                anomaly_flag=False,
                anomaly_score=0.2,
                feature_importance={"feature1": 0.5, "feature2": 0.3},
                timestamp=1234567890.0,
                model_version="1.0.0"
            )
            
            manager.log_prediction(
                user_id="test_user",
                prediction_result=pred_result,
                input_hash="test_hash_123",
                session_id="session_123"
            )
            print("  ✓ Prediction logging working")
            
            # Test model update logging
            manager.log_model_update(
                user_id="admin_user",
                model_id="rul_model_v1",
                old_version="1.0.0",
                new_version="1.1.0"
            )
            print("  ✓ Model update logging working")
            
            # Test configuration change logging
            old_config = {"threshold": 0.5, "batch_size": 32}
            new_config = {"threshold": 0.6, "batch_size": 64}
            
            manager.log_configuration_change(
                user_id="config_admin",
                config_type="MODEL_CONFIG",
                config_id="main_config",
                old_config=old_config,
                new_config=new_config
            )
            print("  ✓ Configuration change logging working")
            
            # Test data access logging
            manager.log_data_access(
                user_id="data_scientist",
                data_type="TRAINING_DATA",
                data_id="es12_dataset",
                access_type="READ"
            )
            print("  ✓ Data access logging working")
            
            print("✓ AuditTrailManager tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error testing AuditTrailManager: {e}")
            return False


def test_compliance_report_generator():
    """Test compliance report generator"""
    print("Testing ComplianceReportGenerator...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            generator = ComplianceReportGenerator(reports_dir=temp_dir)
            
            # Create mock validation results
            validation_results = [
                ValidationResult(
                    protocol_id="TEST_PROTOCOL",
                    test_id="TEST_001",
                    test_name="Algorithm Validation",
                    status="passed",
                    execution_date="2024-01-01T12:00:00Z",
                    executed_by="system",
                    results={"rmse": 5.2, "r2": 0.85},
                    evidence_files=["test_001_evidence.json"]
                ),
                ValidationResult(
                    protocol_id="TEST_PROTOCOL",
                    test_id="TEST_002",
                    test_name="Performance Test",
                    status="failed",
                    execution_date="2024-01-01T12:05:00Z",
                    executed_by="system",
                    results={"response_time": 2.5, "threshold": 1.0},
                    evidence_files=["test_002_evidence.json"]
                )
            ]
            
            # Create mock audit entries
            audit_entries = [
                AuditTrailEntry(
                    entry_id="audit_001",
                    timestamp="2024-01-01T10:00:00Z",
                    user_id="test_user",
                    action="PREDICTION",
                    resource_type="RUL_PREDICTION",
                    resource_id="pred_001",
                    old_value=None,
                    new_value='{"rul_cycles": 50}',
                    ip_address="192.168.1.1",
                    session_id="session_001"
                )
            ]
            
            # Generate compliance report
            report = generator.generate_compliance_report(
                standard=ComplianceStandard.ISO_13485,
                validation_results=validation_results,
                audit_entries=audit_entries,
                reporting_period="Test Period",
                system_version="1.0.0"
            )
            
            assert report.standard == ComplianceStandard.ISO_13485
            assert len(report.validation_results) == 2
            assert report.audit_summary["total_entries"] == 1
            assert report.risk_assessment["overall_risk_level"] in ["low", "medium", "high"]
            assert len(report.recommendations) > 0
            print("  ✓ Compliance report generation working")
            
            # Test HTML report generation
            html_file = generator.generate_html_report(report)
            assert Path(html_file).exists()
            print("  ✓ HTML report generation working")
            
            print("✓ ComplianceReportGenerator tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error testing ComplianceReportGenerator: {e}")
            return False


def test_compliance_manager():
    """Test the main compliance manager"""
    print("Testing ComplianceManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            manager = ComplianceManager(base_dir=temp_dir)
            
            # Test protocol setup for different standards
            iso_protocol = manager.setup_compliance_for_standard(ComplianceStandard.ISO_13485)
            assert iso_protocol.standard == ComplianceStandard.ISO_13485
            print("  ✓ ISO 13485 compliance setup working")
            
            fda_protocol = manager.setup_compliance_for_standard(ComplianceStandard.FDA_21CFR)
            assert fda_protocol.standard == ComplianceStandard.FDA_21CFR
            print("  ✓ FDA 21 CFR compliance setup working")
            
            iso9001_protocol = manager.setup_compliance_for_standard(ComplianceStandard.ISO_9001)
            assert iso9001_protocol.standard == ComplianceStandard.ISO_9001
            print("  ✓ ISO 9001 compliance setup working")
            
            print("✓ ComplianceManager tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error testing ComplianceManager: {e}")
            return False


def test_data_structures():
    """Test compliance data structures"""
    print("Testing compliance data structures...")
    
    try:
        # Test ValidationProtocol
        protocol = ValidationProtocol(
            protocol_id="TEST_PROTOCOL",
            name="Test Protocol",
            standard=ComplianceStandard.ISO_13485,
            version="1.0",
            description="Test protocol description",
            requirements=["req1", "req2"],
            test_procedures=["proc1", "proc2"],
            acceptance_criteria={"metric1": 0.95},
            created_date="2024-01-01T12:00:00Z",
            created_by="test_user"
        )
        assert protocol.protocol_id == "TEST_PROTOCOL"
        assert protocol.standard == ComplianceStandard.ISO_13485
        print("  ✓ ValidationProtocol structure working")
        
        # Test ValidationResult
        result = ValidationResult(
            protocol_id="TEST_PROTOCOL",
            test_id="TEST_001",
            test_name="Test Name",
            status="passed",
            execution_date="2024-01-01T12:00:00Z",
            executed_by="system",
            results={"metric": 0.95},
            evidence_files=["evidence.json"]
        )
        assert result.test_id == "TEST_001"
        assert result.status == "passed"
        print("  ✓ ValidationResult structure working")
        
        # Test AuditTrailEntry
        audit_entry = AuditTrailEntry(
            entry_id="AUDIT_001",
            timestamp="2024-01-01T12:00:00Z",
            user_id="test_user",
            action="TEST_ACTION",
            resource_type="TEST_RESOURCE",
            resource_id="resource_001",
            old_value="old",
            new_value="new",
            ip_address="192.168.1.1",
            session_id="session_001"
        )
        assert audit_entry.entry_id == "AUDIT_001"
        assert audit_entry.action == "TEST_ACTION"
        print("  ✓ AuditTrailEntry structure working")
        
        print("✓ All compliance data structures tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing compliance data structures: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Regulatory Compliance Module")
    print("=" * 60)
    
    all_passed = True
    
    # Test validation protocol manager
    if not test_validation_protocol_manager():
        all_passed = False
    print()
    
    # Test audit trail manager
    if not test_audit_trail_manager():
        all_passed = False
    print()
    
    # Test compliance report generator
    if not test_compliance_report_generator():
        all_passed = False
    print()
    
    # Test compliance manager
    if not test_compliance_manager():
        all_passed = False
    print()
    
    # Test data structures
    if not test_data_structures():
        all_passed = False
    print()
    
    # Final result
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Regulatory Compliance Module is working correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        return 1


if __name__ == "__main__":
    exit(main())