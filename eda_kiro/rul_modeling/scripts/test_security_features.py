#!/usr/bin/env python3
"""
Test script for advanced security features

This script tests the security features including:
- Data encryption and decryption
- Input sanitization and validation
- Authentication and authorization
- Security audit logging
- Secure RUL service
"""

import sys
import os
import numpy as np
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.security_features import (
    SecurityConfig,
    SecurityEvent,
    DataEncryption,
    InputSanitizer,
    AuthenticationManager,
    SecurityAuditLogger,
    SecureRULService,
    SecurityException,
    generate_security_config,
    CRYPTO_AVAILABLE,
    JWT_AVAILABLE
)


def test_data_encryption():
    """Test data encryption functionality"""
    print("Testing DataEncryption...")
    
    if not CRYPTO_AVAILABLE:
        print("⚠️  Cryptography library not available, skipping encryption tests")
        return
    
    encryptor = DataEncryption()
    
    # Test string encryption
    original_text = "This is sensitive RUL prediction data"
    encrypted_data = encryptor.encrypt_data(original_text)
    decrypted_text = encryptor.decrypt_data(encrypted_data, return_type="string")
    
    assert decrypted_text == original_text
    print("✓ String encryption/decryption working")
    
    # Test dictionary encryption
    original_dict = {
        "rul_cycles": 150,
        "degradation_score": 0.3,
        "anomaly_flag": False
    }
    
    encrypted_dict_data = encryptor.encrypt_data(original_dict)
    decrypted_dict = encryptor.decrypt_data(encrypted_dict_data, return_type="json")
    
    assert decrypted_dict == original_dict
    print("✓ Dictionary encryption/decryption working")
    
    # Test file encryption
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_data.txt")
        encrypted_file = os.path.join(temp_dir, "test_data.txt.encrypted")
        decrypted_file = os.path.join(temp_dir, "test_data_decrypted.txt")
        
        # Create test file
        with open(test_file, 'w') as f:
            f.write("Sensitive model parameters and predictions")
        
        # Encrypt file
        result_path = encryptor.encrypt_file(test_file, encrypted_file)
        assert os.path.exists(result_path)
        print("✓ File encryption working")
        
        # Decrypt file
        decrypted_path = encryptor.decrypt_file(encrypted_file, decrypted_file)
        
        # Verify content
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        
        assert decrypted_content == "Sensitive model parameters and predictions"
        print("✓ File decryption working")
    
    # Test key pair generation
    private_key, public_key = encryptor.generate_key_pair()
    assert private_key.startswith(b'-----BEGIN PRIVATE KEY-----')
    assert public_key.startswith(b'-----BEGIN PUBLIC KEY-----')
    print("✓ RSA key pair generation working")
    
    # Test data hashing
    test_data = "data to hash"
    hash_sha256 = encryptor.hash_data(test_data, "sha256")
    hash_sha512 = encryptor.hash_data(test_data, "sha512")
    
    assert len(hash_sha256) == 64  # SHA256 produces 64 hex characters
    assert len(hash_sha512) == 128  # SHA512 produces 128 hex characters
    print("✓ Data hashing working")
    
    print("DataEncryption tests passed!\n")


def test_input_sanitizer():
    """Test input sanitization and validation"""
    print("Testing InputSanitizer...")
    
    config = SecurityConfig()
    sanitizer = InputSanitizer(config)
    
    # Test string sanitization
    clean_string = "This is a normal string"
    sanitized = sanitizer.sanitize_string(clean_string)
    assert sanitized == clean_string
    print("✓ Clean string sanitization working")
    
    # Test HTML escaping
    html_string = "<script>alert('xss')</script>"
    try:
        sanitizer.sanitize_string(html_string)
        assert False, "Should have detected XSS"
    except ValueError as e:
        assert "XSS attack detected" in str(e)
        print("✓ XSS detection working")
    
    # Test SQL injection detection
    sql_string = "'; DROP TABLE users; --"
    try:
        sanitizer.sanitize_string(sql_string)
        assert False, "Should have detected SQL injection"
    except ValueError as e:
        assert "SQL injection detected" in str(e)
        print("✓ SQL injection detection working")
    
    # Test numeric validation
    valid_number = sanitizer.validate_numeric_input("123.45", min_value=0, max_value=1000)
    assert valid_number == 123.45
    print("✓ Numeric validation working")
    
    # Test invalid numeric input
    try:
        sanitizer.validate_numeric_input("not_a_number")
        assert False, "Should have rejected non-numeric input"
    except ValueError:
        print("✓ Invalid numeric input rejection working")
    
    # Test array validation
    valid_array = [1.0, 2.0, 3.0, 4.0, 5.0]
    validated_array = sanitizer.validate_array_input(valid_array, max_size=10)
    assert np.array_equal(validated_array, np.array(valid_array))
    print("✓ Array validation working")
    
    # Test array size limit
    large_array = list(range(20000))  # Exceeds default max_size
    try:
        sanitizer.validate_array_input(large_array, max_size=10000)
        assert False, "Should have rejected large array"
    except ValueError as e:
        assert "Array too large" in str(e)
        print("✓ Array size limit working")
    
    # Test IP address validation
    valid_ip = sanitizer.validate_ip_address("192.168.1.1")
    assert valid_ip == "192.168.1.1"
    print("✓ IP address validation working")
    
    # Test invalid IP address
    try:
        sanitizer.validate_ip_address("999.999.999.999")
        assert False, "Should have rejected invalid IP"
    except ValueError:
        print("✓ Invalid IP address rejection working")
    
    # Test IP whitelist
    config.allowed_ip_ranges = ["192.168.1.0/24", "10.0.0.0/8"]
    sanitizer_with_whitelist = InputSanitizer(config)
    
    assert sanitizer_with_whitelist.check_ip_whitelist("192.168.1.100") == True
    assert sanitizer_with_whitelist.check_ip_whitelist("10.5.5.5") == True
    assert sanitizer_with_whitelist.check_ip_whitelist("8.8.8.8") == False
    print("✓ IP whitelist checking working")
    
    # Test IP blacklist
    config.blocked_ip_addresses = ["192.168.1.100", "10.0.0.1"]
    sanitizer_with_blacklist = InputSanitizer(config)
    
    assert sanitizer_with_blacklist.check_ip_blacklist("192.168.1.100") == True
    assert sanitizer_with_blacklist.check_ip_blacklist("192.168.1.101") == False
    print("✓ IP blacklist checking working")
    
    print("InputSanitizer tests passed!\n")


def test_authentication_manager():
    """Test authentication and authorization"""
    print("Testing AuthenticationManager...")
    
    config = SecurityConfig()
    auth_manager = AuthenticationManager(config)
    
    # Test password hashing and verification
    password = "SecurePassword123!"
    password_hash = auth_manager.hash_password(password)
    
    assert auth_manager.verify_password(password, password_hash) == True
    assert auth_manager.verify_password("wrong_password", password_hash) == False
    print("✓ Password hashing and verification working")
    
    # Test weak password rejection
    try:
        auth_manager.hash_password("weak")
        assert False, "Should have rejected weak password"
    except ValueError as e:
        assert "must be at least" in str(e)
        print("✓ Weak password rejection working")
    
    # Test rate limiting
    user_id = "test_user"
    ip_address = "192.168.1.100"
    
    # Should allow initially
    assert auth_manager.check_rate_limit(user_id, ip_address) == True
    
    # Record multiple failed attempts
    for _ in range(config.max_login_attempts):
        auth_manager.record_failed_login(user_id, ip_address)
    
    # Should be rate limited now
    assert auth_manager.check_rate_limit(user_id, ip_address) == False
    print("✓ Rate limiting working")
    
    # Test clearing failed logins
    auth_manager.clear_failed_logins(user_id, ip_address)
    assert auth_manager.check_rate_limit(user_id, ip_address) == True
    print("✓ Failed login clearing working")
    
    # Test JWT token generation and verification (if available)
    if JWT_AVAILABLE:
        token = auth_manager.generate_jwt_token(
            user_id="test_user",
            tenant_id="test_tenant",
            permissions=["predict", "read"]
        )
        
        payload = auth_manager.verify_jwt_token(token)
        assert payload['user_id'] == "test_user"
        assert payload['tenant_id'] == "test_tenant"
        assert "predict" in payload['permissions']
        print("✓ JWT token generation and verification working")
    else:
        print("⚠️  PyJWT not available, skipping JWT tests")
    
    # Test session management
    session_id = auth_manager.create_session("test_user", "test_tenant")
    assert len(session_id) > 0
    print("✓ Session creation working")
    
    session_data = auth_manager.validate_session(session_id)
    assert session_data is not None
    assert session_data['user_id'] == "test_user"
    print("✓ Session validation working")
    
    # Test session invalidation
    auth_manager.invalidate_session(session_id)
    assert auth_manager.validate_session(session_id) is None
    print("✓ Session invalidation working")
    
    print("AuthenticationManager tests passed!\n")


def test_security_audit_logger():
    """Test security audit logging"""
    print("Testing SecurityAuditLogger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_audit.log")
        audit_logger = SecurityAuditLogger(log_file)
        
        # Test authentication event logging
        audit_logger.log_authentication_event(
            user_id="test_user",
            ip_address="192.168.1.100",
            success=True,
            user_agent="TestAgent/1.0"
        )
        
        assert len(audit_logger.events) == 1
        assert audit_logger.events[0].event_type == "authentication"
        print("✓ Authentication event logging working")
        
        # Test data access event logging
        audit_logger.log_data_access_event(
            user_id="test_user",
            tenant_id="test_tenant",
            ip_address="192.168.1.100",
            endpoint="/api/predict",
            request_data={"cycle_number": 50},
            response_status=200
        )
        
        assert len(audit_logger.events) == 2
        assert audit_logger.events[1].event_type == "data_access"
        print("✓ Data access event logging working")
        
        # Test security violation logging
        audit_logger.log_security_violation(
            violation_type="sql_injection",
            ip_address="192.168.1.100",
            details="Detected SQL injection attempt",
            user_id="test_user"
        )
        
        assert len(audit_logger.events) == 3
        assert audit_logger.events[2].event_type == "security_violation"
        assert audit_logger.events[2].severity == "critical"
        print("✓ Security violation logging working")
        
        # Test compliance report generation
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            report_type="summary"
        )
        
        assert report['summary']['total_events'] == 3
        assert 'authentication' in report['summary']['events_by_type']
        assert 'data_access' in report['summary']['events_by_type']
        assert 'security_violation' in report['summary']['events_by_type']
        print("✓ Compliance report generation working")
        
        # Test audit log export
        export_file = os.path.join(temp_dir, "exported_audit.json")
        audit_logger.export_audit_log(export_file, format="json")
        
        assert os.path.exists(export_file)
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 3
        print("✓ Audit log export working")
    
    print("SecurityAuditLogger tests passed!\n")


def test_secure_rul_service():
    """Test secure RUL service"""
    print("Testing SecureRULService...")
    
    config = SecurityConfig(
        encryption_enabled=CRYPTO_AVAILABLE,
        input_validation_enabled=True,
        audit_logging_enabled=True
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_log = os.path.join(temp_dir, "secure_audit.log")
        service = SecureRULService(config, audit_log)
        
        # Test authentication (create a valid session first)
        if JWT_AVAILABLE:
            token = service.auth_manager.generate_jwt_token(
                user_id="test_user",
                tenant_id="test_tenant",
                permissions=["predict"]
            )
        else:
            token = service.auth_manager.create_session("test_user", "test_tenant")
        
        ip_address = "192.168.1.100"
        user_agent = "TestClient/1.0"
        
        # Test successful authentication
        auth_info = service.authenticate_request(token, ip_address, user_agent)
        assert auth_info['user_id'] == "test_user"
        print("✓ Request authentication working")
        
        # Test input validation
        cycle_data = {
            'cycle_number': 50,
            'capacitor_id': 'C1',
            'timestamp': time.time()
        }
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        validated_cycle_data, validated_features = service.validate_prediction_input(
            cycle_data, features
        )
        
        assert validated_cycle_data['cycle_number'] == 50
        assert len(validated_features) == 5
        print("✓ Input validation working")
        
        # Test secure prediction
        prediction_result = service.secure_predict(
            token=token,
            ip_address=ip_address,
            cycle_data=cycle_data,
            features=features,
            user_agent=user_agent
        )
        
        assert prediction_result.rul_cycles >= 0
        assert prediction_result.model_version == "secure_v1.0"
        print("✓ Secure prediction working")
        
        # Test invalid authentication
        try:
            service.authenticate_request("invalid_token", ip_address, user_agent)
            assert False, "Should have rejected invalid token"
        except SecurityException:
            print("✓ Invalid token rejection working")
        
        # Test malicious input detection
        malicious_cycle_data = {
            'cycle_number': "'; DROP TABLE cycles; --",
            'capacitor_id': '<script>alert("xss")</script>'
        }
        
        try:
            service.validate_prediction_input(malicious_cycle_data, features)
            assert False, "Should have detected malicious input"
        except SecurityException as e:
            assert "validation failed" in str(e)
            print("✓ Malicious input detection working")
        
        # Test security status
        status = service.get_security_status()
        assert 'encryption_enabled' in status
        assert 'input_validation_enabled' in status
        assert 'audit_logging_enabled' in status
        print("✓ Security status reporting working")
    
    print("SecureRULService tests passed!\n")


def test_security_config_generation():
    """Test security configuration generation"""
    print("Testing security configuration generation...")
    
    # Test production config
    prod_config = generate_security_config("production", enable_all_features=True)
    assert prod_config.encryption_enabled == True
    assert prod_config.tls_enabled == True
    assert prod_config.password_min_length == 16
    assert prod_config.session_timeout_minutes == 15
    assert prod_config.max_login_attempts == 3
    print("✓ Production security config generation working")
    
    # Test development config
    dev_config = generate_security_config("development", enable_all_features=False)
    assert dev_config.encryption_enabled == False
    assert dev_config.tls_enabled == False
    assert dev_config.password_min_length == 8
    assert dev_config.session_timeout_minutes == 60
    assert dev_config.max_login_attempts == 10
    print("✓ Development security config generation working")
    
    # Test staging config
    staging_config = generate_security_config("staging", enable_all_features=True)
    assert staging_config.encryption_enabled == True
    assert staging_config.tls_enabled == True
    assert staging_config.mtls_enabled == False  # Different from production
    assert staging_config.password_min_length == 12
    print("✓ Staging security config generation working")
    
    print("Security configuration generation tests passed!\n")


def test_security_edge_cases():
    """Test security edge cases and error handling"""
    print("Testing security edge cases...")
    
    config = SecurityConfig()
    
    # Test input sanitizer with edge cases
    sanitizer = InputSanitizer(config)
    
    # Test empty string
    empty_result = sanitizer.sanitize_string("")
    assert empty_result == ""
    print("✓ Empty string handling working")
    
    # Test very long string
    long_string = "a" * 2000
    try:
        sanitizer.sanitize_string(long_string, max_length=1000)
        assert False, "Should have rejected long string"
    except ValueError as e:
        assert "too long" in str(e)
        print("✓ Long string rejection working")
    
    # Test numeric edge cases
    assert sanitizer.validate_numeric_input("0") == 0.0
    assert sanitizer.validate_numeric_input("-123.45", allow_negative=True) == -123.45
    
    try:
        sanitizer.validate_numeric_input("inf")
        assert False, "Should have rejected infinite value"
    except ValueError:
        print("✓ Infinite value rejection working")
    
    try:
        sanitizer.validate_numeric_input("nan")
        assert False, "Should have rejected NaN value"
    except ValueError:
        print("✓ NaN value rejection working")
    
    # Test array edge cases
    empty_array = sanitizer.validate_array_input([])
    assert len(empty_array) == 0
    print("✓ Empty array handling working")
    
    # Test authentication edge cases
    auth_manager = AuthenticationManager(config)
    
    # Test session timeout
    session_id = auth_manager.create_session("test_user")
    session = auth_manager.active_sessions[session_id]
    
    # Simulate old session
    session['last_activity'] = datetime.now() - timedelta(hours=2)
    
    # Should be invalid due to timeout
    assert auth_manager.validate_session(session_id) is None
    print("✓ Session timeout handling working")
    
    print("Security edge cases tests passed!\n")


def main():
    """Run all security feature tests"""
    print("=" * 60)
    print("ADVANCED SECURITY FEATURES TESTS")
    print("=" * 60)
    
    try:
        test_data_encryption()
        test_input_sanitizer()
        test_authentication_manager()
        test_security_audit_logger()
        test_secure_rul_service()
        test_security_config_generation()
        test_security_edge_cases()
        
        print("=" * 60)
        print("ALL SECURITY FEATURES TESTS PASSED! ✓")
        print("=" * 60)
        
        # Print availability status
        print("\nLibrary Availability Status:")
        print(f"Cryptography: {'✓ Available' if CRYPTO_AVAILABLE else '❌ Not Available'}")
        print(f"PyJWT: {'✓ Available' if JWT_AVAILABLE else '❌ Not Available'}")
        
        if not CRYPTO_AVAILABLE:
            print("\n⚠️  Install cryptography for full encryption features: pip install cryptography")
        if not JWT_AVAILABLE:
            print("⚠️  Install PyJWT for JWT token features: pip install PyJWT")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())