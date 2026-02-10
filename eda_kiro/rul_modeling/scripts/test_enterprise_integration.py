#!/usr/bin/env python3
"""
Test script for enterprise integration capabilities
"""

import asyncio
import logging
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.enterprise_integration import (
    EnterpriseIntegrationManager,
    APIKeyAuthProvider,
    JWTAuthProvider,
    SQLiteConnector,
    AuditLogger,
    MQTTClient,
    OPCUAClient,
    User,
    AuditEventType,
    AuthenticationMethod,
    create_enterprise_integration_system
)
from true_rul.data_structures import PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_user() -> User:
    """Create a test user"""
    return User(
        user_id="test_user_123",
        username="testuser",
        email="test@example.com",
        roles=["operator"],
        permissions=["read_predictions", "create_predictions"]
    )


def create_test_admin_user() -> User:
    """Create a test admin user"""
    return User(
        user_id="admin_user_456",
        username="admin",
        email="admin@example.com",
        roles=["admin"],
        permissions=["read_predictions", "create_predictions", "manage_users", "view_audit"]
    )


def create_test_prediction_result() -> PredictionResult:
    """Create a test prediction result"""
    return PredictionResult(
        rul_cycles=75,
        rul_confidence_lower=65,
        rul_confidence_upper=85,
        degradation_score=0.25,
        degradation_stage="early_degradation",
        anomaly_flag=False,
        anomaly_score=0.15,
        feature_importance={"voltage_drop": 0.4, "response_time": 0.6},
        timestamp=datetime.now(),
        model_version="v2.1.0",
        capacitor_id="CAP_001",
        cycle_number=125
    )


async def test_api_key_authentication():
    """Test API key authentication"""
    logger.info("Testing API key authentication...")
    
    auth_provider = APIKeyAuthProvider()
    test_user = create_test_user()
    
    # Add API key
    api_key = "test_api_key_12345"
    auth_provider.add_api_key(api_key, test_user)
    
    # Test authentication with valid key
    user = await auth_provider.authenticate({"api_key": api_key})
    assert user is not None, "Authentication should succeed with valid API key"
    assert user.user_id == test_user.user_id, "User ID should match"
    
    # Test authentication with invalid key
    user = await auth_provider.authenticate({"api_key": "invalid_key"})
    assert user is None, "Authentication should fail with invalid API key"
    
    # Test token validation
    user = await auth_provider.validate_token(api_key)
    assert user is not None, "Token validation should succeed"
    
    # Remove API key
    auth_provider.remove_api_key(api_key)
    user = await auth_provider.authenticate({"api_key": api_key})
    assert user is None, "Authentication should fail after key removal"
    
    logger.info("✓ API key authentication test passed")


async def test_jwt_authentication():
    """Test JWT authentication"""
    logger.info("Testing JWT authentication...")
    
    secret_key = "test_secret_key_for_jwt"
    auth_provider = JWTAuthProvider(secret_key)
    test_user = create_test_user()
    
    # Add user
    auth_provider.add_user(test_user, "test_password")
    
    # Test authentication with valid credentials
    user = await auth_provider.authenticate({
        "username": "testuser",
        "password": "test_password"
    })
    assert user is not None, "Authentication should succeed with valid credentials"
    assert user.username == "testuser", "Username should match"
    
    # Test authentication with invalid credentials
    user = await auth_provider.authenticate({
        "username": "testuser",
        "password": "wrong_password"
    })
    assert user is None, "Authentication should fail with invalid password"
    
    # Test token creation and validation
    token = auth_provider._create_token(test_user)
    assert token is not None, "Token should be created"
    
    user = await auth_provider.validate_token(token)
    assert user is not None, "Token validation should succeed"
    assert user.username == "testuser", "Username should match from token"
    
    # Test invalid token
    user = await auth_provider.validate_token("invalid.token")
    assert user is None, "Invalid token should not validate"
    
    logger.info("✓ JWT authentication test passed")


async def test_database_connector():
    """Test database connector"""
    logger.info("Testing database connector...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        connector = SQLiteConnector(db_path)
        
        # Test connection
        await connector.connect()
        
        # Test command execution
        success = await connector.execute_command(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
        )
        assert success, "Table creation should succeed"
        
        # Test data insertion
        success = await connector.execute_command(
            "INSERT INTO test_table (name) VALUES (:name)",
            {"name": "test_record"}
        )
        assert success, "Data insertion should succeed"
        
        # Test query execution
        results = await connector.execute_query(
            "SELECT * FROM test_table WHERE name = :name",
            {"name": "test_record"}
        )
        assert len(results) == 1, "Should find one record"
        assert results[0]["name"] == "test_record", "Name should match"
        
        # Test disconnection
        await connector.disconnect()
        
    logger.info("✓ Database connector test passed")


async def test_audit_logging():
    """Test audit logging"""
    logger.info("Testing audit logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "audit_test.db"
        connector = SQLiteConnector(db_path)
        audit_logger = AuditLogger(connector)
        
        await connector.connect()
        
        # Log some events
        await audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            "authentication",
            "login",
            user_id="test_user",
            details={"username": "testuser"},
            ip_address="192.168.1.100"
        )
        
        await audit_logger.log_event(
            AuditEventType.PREDICTION_REQUEST,
            "prediction_api",
            "predict",
            user_id="test_user",
            details={"capacitor_id": "CAP_001", "rul_cycles": 75}
        )
        
        # Query audit events
        events = await audit_logger.get_audit_events(limit=10)
        assert len(events) == 2, f"Should have 2 audit events, got {len(events)}"
        
        # Test filtering by user
        user_events = await audit_logger.get_audit_events(user_id="test_user")
        assert len(user_events) == 2, "Should find 2 events for test_user"
        
        # Test filtering by event type
        login_events = await audit_logger.get_audit_events(event_type=AuditEventType.USER_LOGIN)
        assert len(login_events) == 1, "Should find 1 login event"
        
        await connector.disconnect()
        
    logger.info("✓ Audit logging test passed")


async def test_mqtt_client():
    """Test MQTT client"""
    logger.info("Testing MQTT client...")
    
    # Note: This is a simplified test since we don't have a real MQTT broker
    mqtt_client = MQTTClient("localhost", 1883)
    
    # Test connection (simulated)
    await mqtt_client.connect()
    assert mqtt_client._connected, "MQTT client should be connected"
    
    # Test subscription
    def message_handler(topic: str, message: str):
        logger.info(f"Received message on {topic}: {message}")
    
    await mqtt_client.subscribe("test/topic", message_handler)
    assert "test/topic" in mqtt_client._message_handlers, "Should have message handler"
    
    # Test publishing
    await mqtt_client.publish("test/topic", "test message")
    
    # Test publishing prediction result
    prediction_result = create_test_prediction_result()
    await mqtt_client.publish_prediction_result("predictions/CAP_001", prediction_result)
    
    # Test disconnection
    await mqtt_client.disconnect()
    assert not mqtt_client._connected, "MQTT client should be disconnected"
    
    logger.info("✓ MQTT client test passed")


async def test_opcua_client():
    """Test OPC-UA client"""
    logger.info("Testing OPC-UA client...")
    
    # Note: This is a simplified test since we don't have a real OPC-UA server
    opcua_client = OPCUAClient("opc.tcp://localhost:4840")
    
    # Test connection (simulated)
    await opcua_client.connect()
    assert opcua_client._connected, "OPC-UA client should be connected"
    
    # Test reading node
    value = await opcua_client.read_node("ns=2;i=1001")
    assert value is not None, "Should read a value"
    assert "value" in value, "Value should contain data"
    
    # Test writing node
    await opcua_client.write_node("ns=2;i=1002", 42.5)
    
    # Test subscription
    def change_handler(node_id: str, value: Any):
        logger.info(f"Node {node_id} changed to {value}")
    
    await opcua_client.subscribe_to_changes(["ns=2;i=1001"], change_handler)
    
    # Test disconnection
    await opcua_client.disconnect()
    assert not opcua_client._connected, "OPC-UA client should be disconnected"
    
    logger.info("✓ OPC-UA client test passed")


async def test_enterprise_integration_manager():
    """Test enterprise integration manager"""
    logger.info("Testing enterprise integration manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "enterprise_test.db"
        
        # Create integration system
        integration_manager = create_enterprise_integration_system(
            db_path=db_path,
            auth_method=AuthenticationMethod.API_KEY
        )
        
        # Initialize
        await integration_manager.initialize()
        
        # Set up test user
        test_user = create_test_user()
        api_key = "test_enterprise_key"
        integration_manager.auth_provider.add_api_key(api_key, test_user)
        
        # Test authentication
        user = await integration_manager.authenticate_request({"api_key": api_key})
        assert user is not None, "Authentication should succeed"
        assert user.user_id == test_user.user_id, "User ID should match"
        
        # Test token validation
        user = await integration_manager.validate_token(api_key)
        assert user is not None, "Token validation should succeed"
        
        # Test prediction logging
        prediction_result = create_test_prediction_result()
        await integration_manager.log_prediction_request(
            user=test_user,
            prediction_result=prediction_result,
            processing_time_ms=150.5,
            ip_address="192.168.1.200"
        )
        
        # Test prediction history retrieval
        history = await integration_manager.get_prediction_history(
            user=test_user,
            capacitor_id="CAP_001",
            limit=10
        )
        assert len(history) == 1, "Should have 1 prediction in history"
        assert history[0]["capacitor_id"] == "CAP_001", "Capacitor ID should match"
        
        # Test MQTT integration
        mqtt_client = MQTTClient("localhost", 1883)
        integration_manager.set_mqtt_client(mqtt_client)
        
        # Test OPC-UA integration
        opcua_client = OPCUAClient("opc.tcp://localhost:4840")
        integration_manager.set_opcua_client(opcua_client)
        
        # Test shutdown
        await integration_manager.shutdown()
        
    logger.info("✓ Enterprise integration manager test passed")


async def test_user_permissions():
    """Test user permissions and roles"""
    logger.info("Testing user permissions...")
    
    # Create users with different permissions
    operator_user = User(
        user_id="operator_123",
        username="operator",
        email="operator@example.com",
        roles=["operator"],
        permissions=["read_predictions"]
    )
    
    admin_user = User(
        user_id="admin_456",
        username="admin",
        email="admin@example.com",
        roles=["admin"],
        permissions=["read_predictions", "manage_users", "view_audit"]
    )
    
    # Test permission checks
    assert operator_user.has_permission("read_predictions"), "Operator should have read permission"
    assert not operator_user.has_permission("manage_users"), "Operator should not have manage permission"
    
    assert admin_user.has_permission("read_predictions"), "Admin should have read permission"
    assert admin_user.has_permission("manage_users"), "Admin should have manage permission"
    assert admin_user.has_role("admin"), "Admin should have admin role"
    
    # Test admin override (admin role gives all permissions)
    admin_user_simple = User(
        user_id="admin_789",
        username="admin2",
        email="admin2@example.com",
        roles=["admin"],
        permissions=[]  # No explicit permissions
    )
    
    assert admin_user_simple.has_permission("any_permission"), "Admin role should override permissions"
    
    logger.info("✓ User permissions test passed")


async def test_audit_event_filtering():
    """Test audit event filtering"""
    logger.info("Testing audit event filtering...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "audit_filter_test.db"
        connector = SQLiteConnector(db_path)
        audit_logger = AuditLogger(connector)
        
        await connector.connect()
        
        # Log events with different types and users
        base_time = datetime.now()
        
        await audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            "auth",
            "login",
            user_id="user1",
            details={"timestamp_offset": 0}
        )
        
        await audit_logger.log_event(
            AuditEventType.PREDICTION_REQUEST,
            "api",
            "predict",
            user_id="user1",
            details={"timestamp_offset": 1}
        )
        
        await audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            "auth",
            "login",
            user_id="user2",
            details={"timestamp_offset": 2}
        )
        
        # Test filtering by user
        user1_events = await audit_logger.get_audit_events(user_id="user1")
        assert len(user1_events) == 2, "Should find 2 events for user1"
        
        user2_events = await audit_logger.get_audit_events(user_id="user2")
        assert len(user2_events) == 1, "Should find 1 event for user2"
        
        # Test filtering by event type
        login_events = await audit_logger.get_audit_events(event_type=AuditEventType.USER_LOGIN)
        assert len(login_events) == 2, "Should find 2 login events"
        
        prediction_events = await audit_logger.get_audit_events(event_type=AuditEventType.PREDICTION_REQUEST)
        assert len(prediction_events) == 1, "Should find 1 prediction event"
        
        # Test time filtering
        future_time = base_time + timedelta(hours=1)
        future_events = await audit_logger.get_audit_events(start_time=future_time)
        assert len(future_events) == 0, "Should find no events in the future"
        
        await connector.disconnect()
        
    logger.info("✓ Audit event filtering test passed")


async def main():
    """Run all tests"""
    logger.info("Starting enterprise integration tests...")
    
    try:
        await test_api_key_authentication()
        await test_jwt_authentication()
        await test_database_connector()
        await test_audit_logging()
        await test_mqtt_client()
        await test_opcua_client()
        await test_enterprise_integration_manager()
        await test_user_permissions()
        await test_audit_event_filtering()
        
        logger.info("🎉 All enterprise integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())