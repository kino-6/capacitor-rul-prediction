"""
Regulatory Compliance and Validation Module

This module provides validation protocols for regulated industries,
compliance reporting for quality standards (ISO, FDA, etc.),
audit trails for model decisions, and validation documentation generation.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import uuid
import pickle
import numpy as np
import pandas as pd

from .data_structures import PredictionResult
from .model_evaluator import ModelEvaluator


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    ISO_13485 = "ISO 13485"  # Medical devices
    ISO_9001 = "ISO 9001"    # Quality management
    FDA_21CFR = "FDA 21 CFR Part 820"  # FDA medical device regulations
    IEC_62304 = "IEC 62304"  # Medical device software
    GDPR = "GDPR"           # Data protection
    SOX = "SOX"             # Sarbanes-Oxley
    CUSTOM = "Custom"       # Custom compliance requirements


@dataclass
class ValidationProtocol:
    """Validation protocol definition"""
    protocol_id: str
    name: str
    standard: ComplianceStandard
    version: str
    description: str
    requirements: List[str]
    test_procedures: List[str]
    acceptance_criteria: Dict[str, Any]
    created_date: str
    created_by: str


@dataclass
class ValidationResult:
    """Result of a validation test"""
    protocol_id: str
    test_id: str
    test_name: str
    status: str  # "passed", "failed", "not_applicable"
    execution_date: str
    executed_by: str
    results: Dict[str, Any]
    evidence_files: List[str]
    comments: Optional[str] = None


@dataclass
class AuditTrailEntry:
    """Single audit trail entry"""
    entry_id: str
    timestamp: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    old_value: Optional[str]
    new_value: Optional[str]
    ip_address: Optional[str]
    session_id: Optional[str]
    details: Optional[Dict[str, Any]] = None


@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    standard: ComplianceStandard
    report_date: str
    reporting_period: str
    system_version: str
    validation_results: List[ValidationResult]
    audit_summary: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    generated_by: str
    approved_by: Optional[str] = None
    approval_date: Optional[str] = None


class ValidationProtocolManager:
    """Manages validation protocols for different compliance standards"""
    
    def __init__(self, protocols_dir: str = "compliance/protocols"):
        self.protocols_dir = Path(protocols_dir)
        self.protocols_dir.mkdir(parents=True, exist_ok=True)
        self.protocols: Dict[str, ValidationProtocol] = {}
        self.logger = self._setup_logger()
        self._load_protocols()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation protocols"""
        logger = logging.getLogger("validation_protocol_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.protocols_dir / "validation_protocols.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _load_protocols(self):
        """Load existing validation protocols"""
        for protocol_file in self.protocols_dir.glob("*.json"):
            try:
                with open(protocol_file, 'r') as f:
                    protocol_data = json.load(f)
                    # Convert string back to enum
                    if isinstance(protocol_data.get('standard'), str):
                        for standard in ComplianceStandard:
                            if standard.value == protocol_data['standard']:
                                protocol_data['standard'] = standard
                                break
                    protocol = ValidationProtocol(**protocol_data)
                    self.protocols[protocol.protocol_id] = protocol
            except Exception as e:
                self.logger.error(f"Failed to load protocol {protocol_file}: {e}")
                
    def create_iso_13485_protocol(self) -> ValidationProtocol:
        """Create ISO 13485 validation protocol for medical devices"""
        protocol = ValidationProtocol(
            protocol_id="ISO_13485_RUL_PRED",
            name="ISO 13485 RUL Prediction System Validation",
            standard=ComplianceStandard.ISO_13485,
            version="1.0",
            description="Validation protocol for RUL prediction system under ISO 13485",
            requirements=[
                "Design controls verification",
                "Risk management compliance",
                "Software lifecycle processes",
                "Validation and verification activities",
                "Configuration management",
                "Problem resolution procedures"
            ],
            test_procedures=[
                "Algorithm validation testing",
                "Performance verification testing",
                "Safety and effectiveness testing",
                "Usability validation",
                "Cybersecurity assessment",
                "Clinical evaluation (if applicable)"
            ],
            acceptance_criteria={
                "prediction_accuracy": {"min_r2": 0.8, "max_rmse": 10.0},
                "false_positive_rate": {"max_fpr": 0.05},
                "response_time": {"max_seconds": 1.0},
                "availability": {"min_uptime_pct": 99.5},
                "data_integrity": {"checksum_validation": True},
                "audit_trail": {"complete_logging": True}
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            created_by="system_admin"
        )
        
        self.protocols[protocol.protocol_id] = protocol
        self._save_protocol(protocol)
        return protocol
        
    def create_fda_21cfr_protocol(self) -> ValidationProtocol:
        """Create FDA 21 CFR Part 820 validation protocol"""
        protocol = ValidationProtocol(
            protocol_id="FDA_21CFR_RUL_PRED",
            name="FDA 21 CFR Part 820 RUL Prediction System Validation",
            standard=ComplianceStandard.FDA_21CFR,
            version="1.0",
            description="Validation protocol for RUL prediction system under FDA 21 CFR Part 820",
            requirements=[
                "Design validation requirements",
                "Software validation requirements",
                "Risk analysis documentation",
                "Change control procedures",
                "Corrective and preventive actions",
                "Management responsibility"
            ],
            test_procedures=[
                "Installation qualification (IQ)",
                "Operational qualification (OQ)",
                "Performance qualification (PQ)",
                "Traceability matrix verification",
                "User acceptance testing",
                "Regression testing"
            ],
            acceptance_criteria={
                "clinical_accuracy": {"sensitivity": 0.95, "specificity": 0.95},
                "statistical_validation": {"confidence_interval": 0.95},
                "software_validation": {"code_coverage": 0.90},
                "documentation": {"traceability_complete": True},
                "change_control": {"all_changes_documented": True}
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            created_by="system_admin"
        )
        
        self.protocols[protocol.protocol_id] = protocol
        self._save_protocol(protocol)
        return protocol
        
    def create_iso_9001_protocol(self) -> ValidationProtocol:
        """Create ISO 9001 quality management validation protocol"""
        protocol = ValidationProtocol(
            protocol_id="ISO_9001_RUL_PRED",
            name="ISO 9001 Quality Management System Validation",
            standard=ComplianceStandard.ISO_9001,
            version="1.0",
            description="Validation protocol for quality management compliance",
            requirements=[
                "Quality management system requirements",
                "Management responsibility",
                "Resource management",
                "Product realization",
                "Measurement and improvement"
            ],
            test_procedures=[
                "Process validation testing",
                "Quality metrics verification",
                "Customer satisfaction assessment",
                "Continuous improvement validation",
                "Document control verification"
            ],
            acceptance_criteria={
                "process_capability": {"cpk": 1.33},
                "customer_satisfaction": {"min_score": 4.0},
                "defect_rate": {"max_ppm": 100},
                "on_time_delivery": {"min_pct": 95.0},
                "documentation": {"controlled_documents": True}
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            created_by="system_admin"
        )
        
        self.protocols[protocol.protocol_id] = protocol
        self._save_protocol(protocol)
        return protocol
        
    def _save_protocol(self, protocol: ValidationProtocol):
        """Save validation protocol to file"""
        protocol_file = self.protocols_dir / f"{protocol.protocol_id}.json"
        protocol_dict = asdict(protocol)
        # Convert enum to string for JSON serialization
        protocol_dict['standard'] = protocol.standard.value
        with open(protocol_file, 'w') as f:
            json.dump(protocol_dict, f, indent=2)
            
    def get_protocol(self, protocol_id: str) -> Optional[ValidationProtocol]:
        """Get validation protocol by ID"""
        return self.protocols.get(protocol_id)
        
    def list_protocols(self) -> List[ValidationProtocol]:
        """List all available protocols"""
        return list(self.protocols.values())


class ValidationExecutor:
    """Executes validation tests according to protocols"""
    
    def __init__(self, results_dir: str = "compliance/validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation execution"""
        logger = logging.getLogger("validation_executor")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.results_dir / "validation_execution.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def execute_protocol(self, protocol: ValidationProtocol, 
                        model_path: str, test_data_path: str) -> List[ValidationResult]:
        """Execute validation protocol"""
        self.logger.info(f"Executing validation protocol: {protocol.name}")
        
        results = []
        
        # Execute each test procedure
        for i, procedure in enumerate(protocol.test_procedures):
            test_id = f"{protocol.protocol_id}_TEST_{i+1:03d}"
            
            try:
                if "algorithm validation" in procedure.lower():
                    result = self._execute_algorithm_validation(
                        protocol, test_id, procedure, model_path, test_data_path
                    )
                elif "performance verification" in procedure.lower():
                    result = self._execute_performance_verification(
                        protocol, test_id, procedure, model_path, test_data_path
                    )
                elif "safety and effectiveness" in procedure.lower():
                    result = self._execute_safety_effectiveness(
                        protocol, test_id, procedure, model_path, test_data_path
                    )
                else:
                    result = self._execute_generic_test(
                        protocol, test_id, procedure, model_path, test_data_path
                    )
                    
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute test {test_id}: {e}")
                result = ValidationResult(
                    protocol_id=protocol.protocol_id,
                    test_id=test_id,
                    test_name=procedure,
                    status="failed",
                    execution_date=datetime.now(timezone.utc).isoformat(),
                    executed_by="system",
                    results={"error": str(e)},
                    evidence_files=[],
                    comments=f"Test execution failed: {e}"
                )
                results.append(result)
                
        # Save results
        self._save_validation_results(protocol.protocol_id, results)
        
        return results
        
    def _execute_algorithm_validation(self, protocol: ValidationProtocol, 
                                    test_id: str, test_name: str,
                                    model_path: str, test_data_path: str) -> ValidationResult:
        """Execute algorithm validation test"""
        from .rul_predictor import RULPredictor
        from .data_loader import DataLoader
        
        # Load model and data
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        data_loader = DataLoader()
        test_data = data_loader.load_es12_dataset(test_data_path)
        
        # Run predictions and calculate metrics
        predictions = []
        actuals = []
        
        for cap_id, cap_data in test_data.items():
            for cycle in cap_data.cycles:
                if cycle.cycle_number > 10:
                    pred_result = predictor.predict(cycle.vl_series, cycle.vo_series)
                    predictions.append(pred_result.rul_cycles)
                    actuals.append(cap_data.total_cycles - cycle.cycle_number)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
        
        # Check acceptance criteria
        criteria = protocol.acceptance_criteria.get("prediction_accuracy", {})
        status = "passed"
        
        if "min_r2" in criteria and r2 < criteria["min_r2"]:
            status = "failed"
        if "max_rmse" in criteria and rmse > criteria["max_rmse"]:
            status = "failed"
            
        results = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "n_samples": len(predictions),
            "acceptance_criteria": criteria,
            "meets_criteria": status == "passed"
        }
        
        return ValidationResult(
            protocol_id=protocol.protocol_id,
            test_id=test_id,
            test_name=test_name,
            status=status,
            execution_date=datetime.now(timezone.utc).isoformat(),
            executed_by="system",
            results=results,
            evidence_files=[f"{test_id}_algorithm_validation.json"]
        )
        
    def _execute_performance_verification(self, protocol: ValidationProtocol,
                                        test_id: str, test_name: str,
                                        model_path: str, test_data_path: str) -> ValidationResult:
        """Execute performance verification test"""
        import time
        from .rul_predictor import RULPredictor
        
        # Load model
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        # Test response time
        np.random.seed(42)
        response_times = []
        
        for _ in range(100):
            vl = np.random.randn(100)
            vo = np.random.randn(100)
            
            start_time = time.time()
            predictor.predict(vl, vo)
            response_time = time.time() - start_time
            response_times.append(response_time)
            
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        # Check acceptance criteria
        criteria = protocol.acceptance_criteria.get("response_time", {})
        status = "passed"
        
        if "max_seconds" in criteria and max_response_time > criteria["max_seconds"]:
            status = "failed"
            
        results = {
            "avg_response_time": float(avg_response_time),
            "max_response_time": float(max_response_time),
            "n_tests": len(response_times),
            "acceptance_criteria": criteria,
            "meets_criteria": status == "passed"
        }
        
        return ValidationResult(
            protocol_id=protocol.protocol_id,
            test_id=test_id,
            test_name=test_name,
            status=status,
            execution_date=datetime.now(timezone.utc).isoformat(),
            executed_by="system",
            results=results,
            evidence_files=[f"{test_id}_performance_verification.json"]
        )
        
    def _execute_safety_effectiveness(self, protocol: ValidationProtocol,
                                    test_id: str, test_name: str,
                                    model_path: str, test_data_path: str) -> ValidationResult:
        """Execute safety and effectiveness test"""
        from .rul_predictor import RULPredictor
        from .data_loader import DataLoader
        
        # Load model and data
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        data_loader = DataLoader()
        test_data = data_loader.load_es12_dataset(test_data_path)
        
        # Test FPR on normal cycles
        normal_predictions = []
        
        for cap_id, cap_data in test_data.items():
            for cycle in cap_data.cycles[:10]:  # First 10 cycles are normal
                pred_result = predictor.predict(cycle.vl_series, cycle.vo_series)
                normal_predictions.append(pred_result.anomaly_flag)
        
        false_positives = sum(normal_predictions)
        total_normal = len(normal_predictions)
        fpr = false_positives / total_normal if total_normal > 0 else 0
        
        # Check acceptance criteria
        criteria = protocol.acceptance_criteria.get("false_positive_rate", {})
        status = "passed"
        
        if "max_fpr" in criteria and fpr > criteria["max_fpr"]:
            status = "failed"
            
        results = {
            "false_positive_rate": float(fpr),
            "false_positives": false_positives,
            "total_normal_samples": total_normal,
            "acceptance_criteria": criteria,
            "meets_criteria": status == "passed"
        }
        
        return ValidationResult(
            protocol_id=protocol.protocol_id,
            test_id=test_id,
            test_name=test_name,
            status=status,
            execution_date=datetime.now(timezone.utc).isoformat(),
            executed_by="system",
            results=results,
            evidence_files=[f"{test_id}_safety_effectiveness.json"]
        )
        
    def _execute_generic_test(self, protocol: ValidationProtocol,
                            test_id: str, test_name: str,
                            model_path: str, test_data_path: str) -> ValidationResult:
        """Execute generic validation test"""
        # For tests that don't have specific implementations
        results = {
            "test_type": "generic",
            "status": "manual_review_required",
            "message": "This test requires manual execution and review"
        }
        
        return ValidationResult(
            protocol_id=protocol.protocol_id,
            test_id=test_id,
            test_name=test_name,
            status="not_applicable",
            execution_date=datetime.now(timezone.utc).isoformat(),
            executed_by="system",
            results=results,
            evidence_files=[],
            comments="Manual execution required"
        )
        
    def _save_validation_results(self, protocol_id: str, results: List[ValidationResult]):
        """Save validation results to file"""
        results_file = self.results_dir / f"{protocol_id}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = [asdict(result) for result in results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)


class AuditTrailManager:
    """Manages audit trails for model decisions and updates"""
    
    def __init__(self, audit_dir: str = "compliance/audit_trails"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for audit trails"""
        logger = logging.getLogger("audit_trail_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.audit_dir / "audit_trail.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def log_prediction(self, user_id: str, prediction_result: PredictionResult,
                      input_hash: str, session_id: Optional[str] = None,
                      ip_address: Optional[str] = None):
        """Log a prediction decision"""
        entry = AuditTrailEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action="PREDICTION",
            resource_type="RUL_PREDICTION",
            resource_id=input_hash,
            old_value=None,
            new_value=json.dumps(asdict(prediction_result)),
            ip_address=ip_address,
            session_id=session_id,
            details={
                "rul_cycles": prediction_result.rul_cycles,
                "anomaly_flag": prediction_result.anomaly_flag,
                "degradation_stage": prediction_result.degradation_stage,
                "model_version": prediction_result.model_version
            }
        )
        
        self._save_audit_entry(entry)
        
    def log_model_update(self, user_id: str, model_id: str, 
                        old_version: str, new_version: str,
                        session_id: Optional[str] = None,
                        ip_address: Optional[str] = None):
        """Log a model update"""
        entry = AuditTrailEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action="MODEL_UPDATE",
            resource_type="ML_MODEL",
            resource_id=model_id,
            old_value=old_version,
            new_value=new_version,
            ip_address=ip_address,
            session_id=session_id,
            details={
                "update_type": "version_change",
                "automated": False
            }
        )
        
        self._save_audit_entry(entry)
        
    def log_configuration_change(self, user_id: str, config_type: str,
                                config_id: str, old_config: Dict[str, Any],
                                new_config: Dict[str, Any],
                                session_id: Optional[str] = None,
                                ip_address: Optional[str] = None):
        """Log a configuration change"""
        entry = AuditTrailEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action="CONFIG_CHANGE",
            resource_type=config_type,
            resource_id=config_id,
            old_value=json.dumps(old_config),
            new_value=json.dumps(new_config),
            ip_address=ip_address,
            session_id=session_id,
            details={
                "change_summary": self._summarize_config_changes(old_config, new_config)
            }
        )
        
        self._save_audit_entry(entry)
        
    def log_data_access(self, user_id: str, data_type: str, data_id: str,
                       access_type: str, session_id: Optional[str] = None,
                       ip_address: Optional[str] = None):
        """Log data access"""
        entry = AuditTrailEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action=f"DATA_{access_type.upper()}",
            resource_type=data_type,
            resource_id=data_id,
            old_value=None,
            new_value=None,
            ip_address=ip_address,
            session_id=session_id,
            details={
                "access_type": access_type
            }
        )
        
        self._save_audit_entry(entry)
        
    def _save_audit_entry(self, entry: AuditTrailEntry):
        """Save audit entry to file"""
        # Save to daily log file
        date_str = datetime.now().strftime('%Y%m%d')
        audit_file = self.audit_dir / f"audit_trail_{date_str}.jsonl"
        
        with open(audit_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
            
        self.logger.info(f"Audit entry logged: {entry.action} by {entry.user_id}")
        
    def _summarize_config_changes(self, old_config: Dict[str, Any], 
                                 new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize configuration changes"""
        changes = {}
        
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            
            if old_val != new_val:
                changes[key] = {
                    "old": old_val,
                    "new": new_val
                }
                
        return changes
        
    def get_audit_trail(self, start_date: str, end_date: str,
                       user_id: Optional[str] = None,
                       action: Optional[str] = None) -> List[AuditTrailEntry]:
        """Retrieve audit trail entries"""
        entries = []
        
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Read from daily log files
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime('%Y%m%d')
            audit_file = self.audit_dir / f"audit_trail_{date_str}.jsonl"
            
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    for line in f:
                        try:
                            entry_data = json.loads(line.strip())
                            entry = AuditTrailEntry(**entry_data)
                            
                            entry_dt = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                            
                            if start_dt <= entry_dt <= end_dt:
                                if user_id is None or entry.user_id == user_id:
                                    if action is None or entry.action == action:
                                        entries.append(entry)
                                        
                        except Exception as e:
                            self.logger.error(f"Failed to parse audit entry: {e}")
                            
            current_date = current_date.replace(day=current_date.day + 1)
            
        return entries


class ComplianceReportGenerator:
    """Generates compliance reports for various standards"""
    
    def __init__(self, reports_dir: str = "compliance/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for report generation"""
        logger = logging.getLogger("compliance_report_generator")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.reports_dir / "report_generation.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def generate_compliance_report(self, standard: ComplianceStandard,
                                 validation_results: List[ValidationResult],
                                 audit_entries: List[AuditTrailEntry],
                                 reporting_period: str,
                                 system_version: str) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        # Analyze validation results
        total_tests = len(validation_results)
        passed_tests = sum(1 for r in validation_results if r.status == "passed")
        failed_tests = sum(1 for r in validation_results if r.status == "failed")
        
        # Analyze audit trail
        audit_summary = self._analyze_audit_trail(audit_entries)
        
        # Perform risk assessment
        risk_assessment = self._perform_risk_assessment(validation_results, audit_entries)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, risk_assessment)
        
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            standard=standard,
            report_date=datetime.now(timezone.utc).isoformat(),
            reporting_period=reporting_period,
            system_version=system_version,
            validation_results=validation_results,
            audit_summary=audit_summary,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            generated_by="system"
        )
        
        # Save report
        self._save_report(report)
        
        return report
        
    def _analyze_audit_trail(self, audit_entries: List[AuditTrailEntry]) -> Dict[str, Any]:
        """Analyze audit trail for compliance metrics"""
        if not audit_entries:
            return {"total_entries": 0}
            
        actions = {}
        users = set()
        resources = {}
        
        for entry in audit_entries:
            actions[entry.action] = actions.get(entry.action, 0) + 1
            users.add(entry.user_id)
            resources[entry.resource_type] = resources.get(entry.resource_type, 0) + 1
            
        return {
            "total_entries": len(audit_entries),
            "unique_users": len(users),
            "actions_summary": actions,
            "resources_summary": resources,
            "date_range": {
                "start": min(entry.timestamp for entry in audit_entries),
                "end": max(entry.timestamp for entry in audit_entries)
            }
        }
        
    def _perform_risk_assessment(self, validation_results: List[ValidationResult],
                               audit_entries: List[AuditTrailEntry]) -> Dict[str, Any]:
        """Perform risk assessment based on validation and audit data"""
        risks = []
        
        # Check for failed validations
        failed_validations = [r for r in validation_results if r.status == "failed"]
        if failed_validations:
            risks.append({
                "type": "validation_failure",
                "severity": "high",
                "description": f"{len(failed_validations)} validation tests failed",
                "impact": "System may not meet compliance requirements"
            })
            
        # Check for unusual audit patterns
        if audit_entries:
            config_changes = [e for e in audit_entries if e.action == "CONFIG_CHANGE"]
            if len(config_changes) > 10:  # Arbitrary threshold
                risks.append({
                    "type": "frequent_config_changes",
                    "severity": "medium",
                    "description": f"{len(config_changes)} configuration changes detected",
                    "impact": "Frequent changes may indicate instability"
                })
                
        # Overall risk level
        if any(r["severity"] == "high" for r in risks):
            overall_risk = "high"
        elif any(r["severity"] == "medium" for r in risks):
            overall_risk = "medium"
        else:
            overall_risk = "low"
            
        return {
            "overall_risk_level": overall_risk,
            "identified_risks": risks,
            "risk_count": len(risks)
        }
        
    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check validation results
        failed_validations = [r for r in validation_results if r.status == "failed"]
        if failed_validations:
            recommendations.append(
                "Address failed validation tests before production deployment"
            )
            recommendations.append(
                "Review and update validation protocols if necessary"
            )
            
        # Check risk level
        if risk_assessment["overall_risk_level"] == "high":
            recommendations.append(
                "Implement immediate corrective actions for high-risk items"
            )
            recommendations.append(
                "Increase monitoring and validation frequency"
            )
        elif risk_assessment["overall_risk_level"] == "medium":
            recommendations.append(
                "Monitor identified risks and implement preventive measures"
            )
            
        # General recommendations
        recommendations.extend([
            "Maintain regular compliance monitoring and reporting",
            "Ensure all personnel are trained on compliance requirements",
            "Review and update compliance procedures annually"
        ])
        
        return recommendations
        
    def _save_report(self, report: ComplianceReport):
        """Save compliance report to file"""
        report_file = self.reports_dir / f"{report.standard.value}_{report.report_id}.json"
        
        report_dict = asdict(report)
        # Convert enum to string for JSON serialization
        report_dict['standard'] = report.standard.value
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        self.logger.info(f"Compliance report saved: {report_file}")
        
    def generate_html_report(self, report: ComplianceReport) -> str:
        """Generate HTML version of compliance report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {standard}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .risk-high {{ color: red; font-weight: bold; }}
                .risk-medium {{ color: orange; font-weight: bold; }}
                .risk-low {{ color: green; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Report</h1>
                <p><strong>Standard:</strong> {standard}</p>
                <p><strong>Report Date:</strong> {report_date}</p>
                <p><strong>Reporting Period:</strong> {reporting_period}</p>
                <p><strong>System Version:</strong> {system_version}</p>
            </div>
            
            <div class="section">
                <h2>Validation Results Summary</h2>
                <p>Total Tests: {total_tests}</p>
                <p>Passed: {passed_tests}</p>
                <p>Failed: {failed_tests}</p>
            </div>
            
            <div class="section">
                <h2>Risk Assessment</h2>
                <p>Overall Risk Level: <span class="risk-{risk_level}">{risk_level_upper}</span></p>
                <p>Identified Risks: {risk_count}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {recommendations_html}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Calculate summary statistics
        total_tests = len(report.validation_results)
        passed_tests = sum(1 for r in report.validation_results if r.status == "passed")
        failed_tests = sum(1 for r in report.validation_results if r.status == "failed")
        
        # Format recommendations
        recommendations_html = "\n".join(
            f"<li>{rec}</li>" for rec in report.recommendations
        )
        
        html_content = html_template.format(
            standard=report.standard.value,
            report_date=report.report_date,
            reporting_period=report.reporting_period,
            system_version=report.system_version,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            risk_level=report.risk_assessment["overall_risk_level"],
            risk_level_upper=report.risk_assessment["overall_risk_level"].upper(),
            risk_count=report.risk_assessment["risk_count"],
            recommendations_html=recommendations_html
        )
        
        # Save HTML report
        html_file = self.reports_dir / f"{report.standard.value}_{report.report_id}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
            
        return str(html_file)


class ComplianceManager:
    """Main compliance management interface"""
    
    def __init__(self, base_dir: str = "compliance"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.protocol_manager = ValidationProtocolManager(
            str(self.base_dir / "protocols")
        )
        self.validator = ValidationExecutor(
            str(self.base_dir / "validation_results")
        )
        self.audit_manager = AuditTrailManager(
            str(self.base_dir / "audit_trails")
        )
        self.report_generator = ComplianceReportGenerator(
            str(self.base_dir / "reports")
        )
        
    def setup_compliance_for_standard(self, standard: ComplianceStandard) -> ValidationProtocol:
        """Setup compliance for a specific standard"""
        if standard == ComplianceStandard.ISO_13485:
            return self.protocol_manager.create_iso_13485_protocol()
        elif standard == ComplianceStandard.FDA_21CFR:
            return self.protocol_manager.create_fda_21cfr_protocol()
        elif standard == ComplianceStandard.ISO_9001:
            return self.protocol_manager.create_iso_9001_protocol()
        else:
            raise ValueError(f"Unsupported compliance standard: {standard}")
            
    def run_compliance_validation(self, standard: ComplianceStandard,
                                model_path: str, test_data_path: str) -> ComplianceReport:
        """Run complete compliance validation"""
        # Get or create protocol
        protocols = [p for p in self.protocol_manager.list_protocols() 
                    if p.standard == standard]
        
        if not protocols:
            protocol = self.setup_compliance_for_standard(standard)
        else:
            protocol = protocols[0]
            
        # Execute validation
        validation_results = self.validator.execute_protocol(
            protocol, model_path, test_data_path
        )
        
        # Get recent audit entries
        end_date = datetime.now(timezone.utc).isoformat()
        start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=30)).isoformat()
        audit_entries = self.audit_manager.get_audit_trail(start_date, end_date)
        
        # Generate compliance report
        report = self.report_generator.generate_compliance_report(
            standard=standard,
            validation_results=validation_results,
            audit_entries=audit_entries,
            reporting_period="Last 30 days",
            system_version="1.0.0"
        )
        
        return report