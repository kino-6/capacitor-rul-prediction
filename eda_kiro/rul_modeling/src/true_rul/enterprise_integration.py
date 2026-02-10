"""
Enterprise Integration Capabilities

This module provides enterprise-grade integration capabilities including
MQTT/OPC-UA protocols, database connectors, authentication, and audit logging.
"""

import asyncio
import json
import logging
import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import uuid
import sqlite3

from .data_structures import CycleData, PredictionResult

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    JWT = "jwt"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class AuditEventType(Enum):
    """Audit event types"""
    PREDICTION_REQUEST = "prediction_request"
    MODEL_UPDATE = "model_update"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    SYSTEM_ERROR = "system_error"


@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission"""
        return permission in self.permissions or "admin" in self.roles
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles


@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    timestamp: datetime
    resource: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "error_message": self.error_message
        }


class AuthenticationProvider(ABC):
    """Abstract base class for authentication providers"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with credentials"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token"""
        pass


class APIKeyAuthProvider(AuthenticationProvider):
    """API Key authentication provider"""
    
    def __init__(self):
        self._api_keys: Dict[str, User] = {}
        
    def add_api_key(self, api_key: str, user: User):
        """Add an API key for a user"""
        self._api_keys[api_key] = user
        
    def remove_api_key(self, api_key: str):
        """Remove an API key"""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate using API key"""
        api_key = credentials.get("api_key")
        if not api_key:
            return None
            
        return self._api_keys.get(api_key)
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate API key token"""
        return self._api_keys.get(token)


class JWTAuthProvider(AuthenticationProvider):
    """JWT authentication provider (simplified implementation)"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self._users: Dict[str, User] = {}
        
    def add_user(self, user: User, password: str):
        """Add a user with password"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        self._users[user.username] = (user, password_hash)
        # In a real implementation, store password hash securely
        
    def _create_token(self, user: User) -> str:
        """Create a simple JWT-like token"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "exp": int((datetime.now() + timedelta(hours=24)).timestamp())
        }
        
        # Simplified token creation (use proper JWT library in production)
        token_data = json.dumps(payload)
        signature = hmac.new(
            self.secret_key.encode(),
            token_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{token_data}.{signature}"
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a token"""
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
                
            token_data, signature = parts
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
                
            payload = json.loads(token_data)
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
                
            return payload
            
        except Exception:
            return None
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate using username/password"""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return None
            
        user_data = self._users.get(username)
        if not user_data:
            return None
            
        user, stored_password_hash = user_data
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if not hmac.compare_digest(password_hash, stored_password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        return user
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token"""
        payload = self._verify_token(token)
        if not payload:
            return None
            
        username = payload.get("username")
        user_data = self._users.get(username)
        return user_data[0] if user_data else None


class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    @abstractmethod
    async def connect(self):
        """Connect to database"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from database"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a command (INSERT, UPDATE, DELETE)"""
        pass


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        
    async def connect(self):
        """Connect to SQLite database"""
        with self._lock:
            if self.connection is None:
                self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self.connection.row_factory = sqlite3.Row
                
                # Create tables if they don't exist
                await self._create_tables()
                
    async def disconnect(self):
        """Disconnect from database"""
        with self._lock:
            if self.connection:
                self.connection.close()
                self.connection = None
                
    async def _create_tables(self):
        """Create necessary tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                capacitor_id TEXT,
                cycle_number INTEGER,
                rul_cycles INTEGER,
                degradation_score REAL,
                anomaly_flag INTEGER,
                model_version TEXT,
                processing_time_ms REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                resource TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success INTEGER,
                error_message TEXT
            )
            """
        ]
        
        for table_sql in tables:
            self.connection.execute(table_sql)
        self.connection.commit()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        with self._lock:
            if not self.connection:
                await self.connect()
                
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a command"""
        try:
            with self._lock:
                if not self.connection:
                    await self.connect()
                    
                cursor = self.connection.cursor()
                
                if params:
                    cursor.execute(command, params)
                else:
                    cursor.execute(command)
                    
                self.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Database command failed: {e}")
            return False


class AuditLogger:
    """Audit logging system"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        
    async def log_event(self, 
                       event_type: AuditEventType,
                       resource: str,
                       action: str,
                       user_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       success: bool = True,
                       error_message: Optional[str] = None):
        """Log an audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            timestamp=datetime.now(),
            resource=resource,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        # Log to database
        await self._log_to_database(event)
        
        # Log to file
        logger.info(f"AUDIT: {event.event_type.value} - {event.action} on {event.resource} by {event.user_id}")
        
    async def _log_to_database(self, event: AuditEvent):
        """Log event to database"""
        try:
            command = """
                INSERT INTO audit_events (
                    event_id, event_type, user_id, timestamp, resource, action,
                    details, ip_address, user_agent, success, error_message
                ) VALUES (
                    :event_id, :event_type, :user_id, :timestamp, :resource, :action,
                    :details, :ip_address, :user_agent, :success, :error_message
                )
            """
            
            params = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "timestamp": event.timestamp.isoformat(),
                "resource": event.resource,
                "action": event.action,
                "details": json.dumps(event.details),
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "success": 1 if event.success else 0,
                "error_message": event.error_message
            }
            
            await self.db_connector.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"Failed to log audit event to database: {e}")
    
    async def get_audit_events(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              user_id: Optional[str] = None,
                              event_type: Optional[AuditEventType] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit events with filtering"""
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = {}
        
        if start_time:
            query += " AND timestamp >= :start_time"
            params["start_time"] = start_time.isoformat()
            
        if end_time:
            query += " AND timestamp <= :end_time"
            params["end_time"] = end_time.isoformat()
            
        if user_id:
            query += " AND user_id = :user_id"
            params["user_id"] = user_id
            
        if event_type:
            query += " AND event_type = :event_type"
            params["event_type"] = event_type.value
            
        query += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit
        
        return await self.db_connector.execute_query(query, params)


class MQTTClient:
    """MQTT client for industrial IoT integration"""
    
    def __init__(self, broker_host: str, broker_port: int = 1883,
                 username: Optional[str] = None, password: Optional[str] = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client = None
        self._connected = False
        self._message_handlers: Dict[str, Callable] = {}
        
    async def connect(self):
        """Connect to MQTT broker"""
        try:
            # Note: This is a simplified implementation
            # In production, use a proper MQTT library like paho-mqtt
            logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
            self._connected = True
            logger.info("Connected to MQTT broker")
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self._connected:
            logger.info("Disconnecting from MQTT broker")
            self._connected = False
            
    async def subscribe(self, topic: str, handler: Callable[[str, str], None]):
        """Subscribe to a topic"""
        if not self._connected:
            await self.connect()
            
        self._message_handlers[topic] = handler
        logger.info(f"Subscribed to MQTT topic: {topic}")
        
    async def publish(self, topic: str, message: str, qos: int = 0):
        """Publish a message to a topic"""
        if not self._connected:
            await self.connect()
            
        logger.info(f"Publishing to MQTT topic {topic}: {message[:100]}...")
        
    async def publish_prediction_result(self, topic: str, result: PredictionResult):
        """Publish a prediction result"""
        message = json.dumps(result.to_dict())
        await self.publish(topic, message)


class OPCUAClient:
    """OPC-UA client for industrial automation integration"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = None
        self._connected = False
        
    async def connect(self):
        """Connect to OPC-UA server"""
        try:
            # Note: This is a simplified implementation
            # In production, use a proper OPC-UA library like asyncua
            logger.info(f"Connecting to OPC-UA server at {self.server_url}")
            self._connected = True
            logger.info("Connected to OPC-UA server")
            
        except Exception as e:
            logger.error(f"Failed to connect to OPC-UA server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from OPC-UA server"""
        if self._connected:
            logger.info("Disconnecting from OPC-UA server")
            self._connected = False
            
    async def read_node(self, node_id: str) -> Any:
        """Read a value from an OPC-UA node"""
        if not self._connected:
            await self.connect()
            
        # Simulate reading a value
        logger.info(f"Reading OPC-UA node: {node_id}")
        return {"value": 42.0, "timestamp": datetime.now()}
    
    async def write_node(self, node_id: str, value: Any):
        """Write a value to an OPC-UA node"""
        if not self._connected:
            await self.connect()
            
        logger.info(f"Writing to OPC-UA node {node_id}: {value}")
    
    async def subscribe_to_changes(self, node_ids: List[str], handler: Callable):
        """Subscribe to node value changes"""
        if not self._connected:
            await self.connect()
            
        logger.info(f"Subscribing to OPC-UA nodes: {node_ids}")


class EnterpriseIntegrationManager:
    """Main enterprise integration manager"""
    
    def __init__(self, 
                 auth_provider: AuthenticationProvider,
                 db_connector: DatabaseConnector,
                 audit_logger: AuditLogger):
        self.auth_provider = auth_provider
        self.db_connector = db_connector
        self.audit_logger = audit_logger
        
        self.mqtt_client: Optional[MQTTClient] = None
        self.opcua_client: Optional[OPCUAClient] = None
        
        self._prediction_handlers: List[Callable] = []
        
    async def initialize(self):
        """Initialize the integration manager"""
        await self.db_connector.connect()
        logger.info("Enterprise integration manager initialized")
        
    async def shutdown(self):
        """Shutdown the integration manager"""
        if self.mqtt_client:
            await self.mqtt_client.disconnect()
            
        if self.opcua_client:
            await self.opcua_client.disconnect()
            
        await self.db_connector.disconnect()
        logger.info("Enterprise integration manager shutdown")
        
    def set_mqtt_client(self, mqtt_client: MQTTClient):
        """Set MQTT client"""
        self.mqtt_client = mqtt_client
        
    def set_opcua_client(self, opcua_client: OPCUAClient):
        """Set OPC-UA client"""
        self.opcua_client = opcua_client
        
    async def authenticate_request(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate a request"""
        try:
            user = await self.auth_provider.authenticate(credentials)
            
            if user:
                await self.audit_logger.log_event(
                    AuditEventType.USER_LOGIN,
                    "authentication",
                    "login",
                    user_id=user.user_id,
                    details={"username": user.username}
                )
            else:
                await self.audit_logger.log_event(
                    AuditEventType.USER_LOGIN,
                    "authentication",
                    "login_failed",
                    details={"credentials": list(credentials.keys())},
                    success=False
                )
                
            return user
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await self.audit_logger.log_event(
                AuditEventType.SYSTEM_ERROR,
                "authentication",
                "authentication_error",
                error_message=str(e),
                success=False
            )
            return None
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate an authentication token"""
        try:
            return await self.auth_provider.validate_token(token)
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    async def log_prediction_request(self, 
                                   user: User,
                                   prediction_result: PredictionResult,
                                   processing_time_ms: float,
                                   ip_address: Optional[str] = None):
        """Log a prediction request"""
        
        # Log to audit system
        await self.audit_logger.log_event(
            AuditEventType.PREDICTION_REQUEST,
            "prediction_api",
            "predict",
            user_id=user.user_id,
            details={
                "capacitor_id": prediction_result.capacitor_id,
                "cycle_number": prediction_result.cycle_number,
                "rul_cycles": prediction_result.rul_cycles,
                "anomaly_flag": prediction_result.anomaly_flag,
                "processing_time_ms": processing_time_ms
            },
            ip_address=ip_address
        )
        
        # Store prediction in database
        await self._store_prediction(user, prediction_result, processing_time_ms)
        
        # Publish to MQTT if configured
        if self.mqtt_client:
            topic = f"rul_predictions/{prediction_result.capacitor_id}"
            await self.mqtt_client.publish_prediction_result(topic, prediction_result)
            
    async def _store_prediction(self, 
                               user: User,
                               prediction_result: PredictionResult,
                               processing_time_ms: float):
        """Store prediction in database"""
        try:
            command = """
                INSERT INTO predictions (
                    id, timestamp, user_id, capacitor_id, cycle_number,
                    rul_cycles, degradation_score, anomaly_flag, model_version, processing_time_ms
                ) VALUES (
                    :id, :timestamp, :user_id, :capacitor_id, :cycle_number,
                    :rul_cycles, :degradation_score, :anomaly_flag, :model_version, :processing_time_ms
                )
            """
            
            params = {
                "id": str(uuid.uuid4()),
                "timestamp": prediction_result.timestamp.isoformat(),
                "user_id": user.user_id,
                "capacitor_id": prediction_result.capacitor_id,
                "cycle_number": prediction_result.cycle_number,
                "rul_cycles": prediction_result.rul_cycles,
                "degradation_score": prediction_result.degradation_score,
                "anomaly_flag": 1 if prediction_result.anomaly_flag else 0,
                "model_version": prediction_result.model_version,
                "processing_time_ms": processing_time_ms
            }
            
            await self.db_connector.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    async def get_prediction_history(self, 
                                   user: User,
                                   capacitor_id: Optional[str] = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history"""
        
        # Check permissions
        if not user.has_permission("read_predictions"):
            raise PermissionError("User does not have permission to read predictions")
            
        query = "SELECT * FROM predictions WHERE 1=1"
        params = {}
        
        if capacitor_id:
            query += " AND capacitor_id = :capacitor_id"
            params["capacitor_id"] = capacitor_id
            
        if start_time:
            query += " AND timestamp >= :start_time"
            params["start_time"] = start_time.isoformat()
            
        if end_time:
            query += " AND timestamp <= :end_time"
            params["end_time"] = end_time.isoformat()
            
        query += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit
        
        results = await self.db_connector.execute_query(query, params)
        
        # Log data access
        await self.audit_logger.log_event(
            AuditEventType.DATA_ACCESS,
            "predictions",
            "query",
            user_id=user.user_id,
            details={
                "capacitor_id": capacitor_id,
                "result_count": len(results)
            }
        )
        
        return results


def create_enterprise_integration_system(
    db_path: Path,
    auth_method: AuthenticationMethod = AuthenticationMethod.API_KEY,
    secret_key: Optional[str] = None
) -> EnterpriseIntegrationManager:
    """
    Create an enterprise integration system
    
    Args:
        db_path: Path to SQLite database
        auth_method: Authentication method to use
        secret_key: Secret key for JWT authentication
        
    Returns:
        Configured EnterpriseIntegrationManager
    """
    
    # Create database connector
    db_connector = SQLiteConnector(db_path)
    
    # Create authentication provider
    if auth_method == AuthenticationMethod.API_KEY:
        auth_provider = APIKeyAuthProvider()
    elif auth_method == AuthenticationMethod.JWT:
        if not secret_key:
            secret_key = "default_secret_key_change_in_production"
        auth_provider = JWTAuthProvider(secret_key)
    else:
        raise ValueError(f"Unsupported authentication method: {auth_method}")
    
    # Create audit logger
    audit_logger = AuditLogger(db_connector)
    
    # Create integration manager
    integration_manager = EnterpriseIntegrationManager(
        auth_provider=auth_provider,
        db_connector=db_connector,
        audit_logger=audit_logger
    )
    
    return integration_manager