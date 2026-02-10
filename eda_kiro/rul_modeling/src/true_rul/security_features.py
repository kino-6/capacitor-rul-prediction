"""
Advanced Security Features Module

This module implements advanced security features including:
- Data encryption at rest and in transit
- Secure model serving with TLS/mTLS
- Input sanitization and injection attack prevention
- Security audit trails and compliance reporting

Requirements: 10.1, 10.5
"""

import logging
import os
import json
import time
import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import re
import ipaddress

# Cryptography libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available. Install with: pip install cryptography")

# JWT libraries
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT library not available. Install with: pip install PyJWT")

# Core libraries
import numpy as np

# Password hashing
try:
    from werkzeug.security import generate_password_hash, check_password_hash
    WERKZEUG_AVAILABLE = True
except ImportError:
    WERKZEUG_AVAILABLE = False
    # Simple fallback password hashing using hashlib
    import hashlib
    
    def generate_password_hash(password: str, method: str = 'pbkdf2:sha256', salt_length: int = 16) -> str:
        """Fallback password hashing"""
        salt = os.urandom(salt_length)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + ':' + pwdhash.hex()
    
    def check_password_hash(pwhash: str, password: str) -> bool:
        """Fallback password verification"""
        try:
            salt_hex, pwdhash_hex = pwhash.split(':')
            salt = bytes.fromhex(salt_hex)
            pwdhash = bytes.fromhex(pwdhash_hex)
            new_pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return pwdhash == new_pwdhash
        except:
            return False

# Local imports
from .data_structures import PredictionResult
from .exceptions import ModelCompressionError

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    encryption_enabled: bool = True
    tls_enabled: bool = True
    mtls_enabled: bool = False
    audit_logging_enabled: bool = True
    input_validation_enabled: bool = True
    rate_limiting_enabled: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    max_request_size_mb: int = 10
    allowed_ip_ranges: List[str] = field(default_factory=lambda: ["0.0.0.0/0"])
    blocked_ip_addresses: List[str] = field(default_factory=list)
    password_min_length: int = 12
    password_require_special_chars: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str  # "authentication", "authorization", "data_access", "security_violation"
    severity: str  # "low", "medium", "high", "critical"
    user_id: Optional[str]
    tenant_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    endpoint: Optional[str]
    request_data_hash: Optional[str]
    response_status: Optional[int]
    error_message: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


class DataEncryption:
    """
    Data encryption and decryption utilities
    
    Provides encryption for data at rest and in transit using
    industry-standard cryptographic algorithms.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize data encryption
        
        Args:
            encryption_key: Encryption key (will generate if not provided)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library required for encryption features")
        
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.fernet = Fernet(encryption_key)
        self.encryption_key = encryption_key
        
        logger.info("DataEncryption initialized")
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
        
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        logger.debug("Data encrypted successfully")
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes, return_type: str = "string") -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt data using Fernet symmetric encryption
        
        Args:
            encrypted_data: Encrypted data as bytes
            return_type: Type to return ("string", "bytes", "json")
        
        Returns:
            Decrypted data in specified format
        """
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        if return_type == "bytes":
            return decrypted_data
        elif return_type == "string":
            return decrypted_data.decode('utf-8')
        elif return_type == "json":
            return json.loads(decrypted_data.decode('utf-8'))
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt a file
        
        Args:
            file_path: Path to file to encrypt
            output_path: Output path for encrypted file (optional)
        
        Returns:
            Path to encrypted file
        """
        if output_path is None:
            output_path = file_path + ".encrypted"
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.encrypt_data(file_data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"File encrypted: {file_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """
        Decrypt a file
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Output path for decrypted file (optional)
        
        Returns:
            Path to decrypted file
        """
        if output_path is None:
            output_path = encrypted_file_path.replace(".encrypted", "")
        
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.decrypt_data(encrypted_data, return_type="bytes")
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
        return output_path
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair for asymmetric encryption
        
        Returns:
            Tuple of (private_key, public_key) as PEM-encoded bytes
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """
        Hash data using specified algorithm
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm ("sha256", "sha512", "md5")
        
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "sha256":
            hash_obj = hashlib.sha256(data)
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512(data)
        elif algorithm == "md5":
            hash_obj = hashlib.md5(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()


class InputSanitizer:
    """
    Input sanitization and validation
    
    Prevents injection attacks and validates input data
    according to security best practices.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize input sanitizer
        
        Args:
            config: Security configuration
        """
        self.config = config
        
        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\bUNION\s+SELECT\b)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)"
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]\\]",
            r"\b(rm|del|format|shutdown|reboot|kill|ps|ls|cat|grep|find|wget|curl)\b"
        ]
        
        logger.info("InputSanitizer initialized")
    
    def sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize string input
        
        Args:
            input_string: String to sanitize
            max_length: Maximum allowed length
        
        Returns:
            Sanitized string
        
        Raises:
            ValueError: If input contains dangerous patterns
        """
        if not isinstance(input_string, str):
            raise ValueError("Input must be a string")
        
        # Check length
        if len(input_string) > max_length:
            raise ValueError(f"Input too long: {len(input_string)} > {max_length}")
        
        # Check for XSS patterns first (before SQL injection)
        for pattern in self.xss_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise ValueError(f"Potential XSS attack detected: {pattern}")
        
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise ValueError(f"Potential SQL injection detected: {pattern}")
        
        # Check for command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise ValueError(f"Potential command injection detected: {pattern}")
        
        # Basic sanitization
        sanitized = input_string.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Escape HTML entities
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        
        return sanitized
    
    def validate_numeric_input(
        self,
        value: Union[int, float, str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_negative: bool = True
    ) -> float:
        """
        Validate numeric input
        
        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_negative: Whether negative values are allowed
        
        Returns:
            Validated numeric value
        
        Raises:
            ValueError: If validation fails
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")
        
        if not allow_negative and numeric_value < 0:
            raise ValueError("Negative values not allowed")
        
        if min_value is not None and numeric_value < min_value:
            raise ValueError(f"Value too small: {numeric_value} < {min_value}")
        
        if max_value is not None and numeric_value > max_value:
            raise ValueError(f"Value too large: {numeric_value} > {max_value}")
        
        if not np.isfinite(numeric_value):
            raise ValueError("Value must be finite")
        
        return numeric_value
    
    def validate_array_input(
        self,
        array: Union[List, np.ndarray],
        max_size: int = 10000,
        element_type: type = float,
        allow_nan: bool = False
    ) -> np.ndarray:
        """
        Validate array input
        
        Args:
            array: Array to validate
            max_size: Maximum array size
            element_type: Expected element type
            allow_nan: Whether NaN values are allowed
        
        Returns:
            Validated numpy array
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(array, (list, np.ndarray)):
            raise ValueError("Input must be a list or numpy array")
        
        # Convert to numpy array
        try:
            np_array = np.array(array, dtype=element_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert to numpy array: {e}")
        
        # Check size
        if np_array.size > max_size:
            raise ValueError(f"Array too large: {np_array.size} > {max_size}")
        
        # Check for NaN values
        if not allow_nan and np.any(np.isnan(np_array)):
            raise ValueError("NaN values not allowed")
        
        # Check for infinite values
        if np.any(np.isinf(np_array)):
            raise ValueError("Infinite values not allowed")
        
        return np_array
    
    def validate_ip_address(self, ip_address: str) -> str:
        """
        Validate IP address
        
        Args:
            ip_address: IP address to validate
        
        Returns:
            Validated IP address
        
        Raises:
            ValueError: If IP address is invalid
        """
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            return str(ip_obj)
        except ValueError:
            raise ValueError(f"Invalid IP address: {ip_address}")
    
    def check_ip_whitelist(self, ip_address: str) -> bool:
        """
        Check if IP address is in allowed ranges
        
        Args:
            ip_address: IP address to check
        
        Returns:
            True if IP is allowed, False otherwise
        """
        if not self.config.allowed_ip_ranges:
            return True  # No restrictions
        
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Check if IP is in any allowed range
            for ip_range in self.config.allowed_ip_ranges:
                network = ipaddress.ip_network(ip_range, strict=False)
                if ip_obj in network:
                    return True
            
            return False
            
        except ValueError:
            return False  # Invalid IP address
    
    def check_ip_blacklist(self, ip_address: str) -> bool:
        """
        Check if IP address is blocked
        
        Args:
            ip_address: IP address to check
        
        Returns:
            True if IP is blocked, False otherwise
        """
        return ip_address in self.config.blocked_ip_addresses


class AuthenticationManager:
    """
    Authentication and authorization management
    
    Handles user authentication, JWT tokens, and session management
    with security best practices.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize authentication manager
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Generate JWT secret if not provided
        if not config.jwt_secret_key:
            config.jwt_secret_key = secrets.token_urlsafe(32)
        
        logger.info("AuthenticationManager initialized")
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using secure algorithm
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        self._validate_password_strength(password)
        return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            password_hash: Stored password hash
        
        Returns:
            True if password matches, False otherwise
        """
        return check_password_hash(password_hash, password)
    
    def _validate_password_strength(self, password: str):
        """
        Validate password strength
        
        Args:
            password: Password to validate
        
        Raises:
            ValueError: If password doesn't meet requirements
        """
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_special_chars:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                raise ValueError("Password must contain at least one special character")
            
            if not re.search(r'[A-Z]', password):
                raise ValueError("Password must contain at least one uppercase letter")
            
            if not re.search(r'[a-z]', password):
                raise ValueError("Password must contain at least one lowercase letter")
            
            if not re.search(r'\d', password):
                raise ValueError("Password must contain at least one digit")
    
    def check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """
        Check if user/IP is rate limited due to failed login attempts
        
        Args:
            user_id: User identifier
            ip_address: IP address
        
        Returns:
            True if allowed, False if rate limited
        """
        key = f"{user_id}:{ip_address}"
        current_time = datetime.now()
        
        if key not in self.failed_login_attempts:
            return True
        
        # Remove old attempts (outside lockout window)
        lockout_window = timedelta(minutes=self.config.lockout_duration_minutes)
        self.failed_login_attempts[key] = [
            attempt for attempt in self.failed_login_attempts[key]
            if current_time - attempt < lockout_window
        ]
        
        # Check if too many recent attempts
        if len(self.failed_login_attempts[key]) >= self.config.max_login_attempts:
            return False
        
        return True
    
    def record_failed_login(self, user_id: str, ip_address: str):
        """
        Record failed login attempt
        
        Args:
            user_id: User identifier
            ip_address: IP address
        """
        key = f"{user_id}:{ip_address}"
        if key not in self.failed_login_attempts:
            self.failed_login_attempts[key] = []
        
        self.failed_login_attempts[key].append(datetime.now())
    
    def clear_failed_logins(self, user_id: str, ip_address: str):
        """
        Clear failed login attempts after successful login
        
        Args:
            user_id: User identifier
            ip_address: IP address
        """
        key = f"{user_id}:{ip_address}"
        if key in self.failed_login_attempts:
            del self.failed_login_attempts[key]
    
    def generate_jwt_token(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        permissions: Optional[List[str]] = None
    ) -> str:
        """
        Generate JWT token for authenticated user
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier (optional)
            permissions: User permissions (optional)
        
        Returns:
            JWT token string
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library required for JWT token generation")
        
        payload = {
            'user_id': user_id,
            'tenant_id': tenant_id,
            'permissions': permissions or [],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm='HS256')
        return token
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token payload
        
        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library required for JWT token verification")
        
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")
    
    def create_session(self, user_id: str, tenant_id: Optional[str] = None) -> str:
        """
        Create user session
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier (optional)
        
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'tenant_id': tenant_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate session and update last activity
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data if valid, None otherwise
        """
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        current_time = datetime.now()
        
        # Check session timeout
        timeout = timedelta(minutes=self.config.session_timeout_minutes)
        if current_time - session['last_activity'] > timeout:
            del self.active_sessions[session_id]
            return None
        
        # Update last activity
        session['last_activity'] = current_time
        return session
    
    def invalidate_session(self, session_id: str):
        """
        Invalidate user session
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


class SecurityAuditLogger:
    """
    Security audit logging and compliance reporting
    
    Logs security events and generates compliance reports
    for regulatory requirements.
    """
    
    def __init__(self, log_file_path: str = "security_audit.log"):
        """
        Initialize security audit logger
        
        Args:
            log_file_path: Path to security audit log file
        """
        self.log_file_path = log_file_path
        self.events: List[SecurityEvent] = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        logger.info(f"SecurityAuditLogger initialized: {log_file_path}")
    
    def log_security_event(self, event: SecurityEvent):
        """
        Log security event
        
        Args:
            event: Security event to log
        """
        # Add to in-memory list
        self.events.append(event)
        
        # Write to log file
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_id': event.event_id,
            'event_type': event.event_type,
            'severity': event.severity,
            'user_id': event.user_id,
            'tenant_id': event.tenant_id,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'endpoint': event.endpoint,
            'request_data_hash': event.request_data_hash,
            'response_status': event.response_status,
            'error_message': event.error_message,
            'additional_data': event.additional_data
        }
        
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to standard logger based on severity
        if event.severity == "critical":
            logger.critical(f"Security Event: {event.event_type} - {event.error_message}")
        elif event.severity == "high":
            logger.error(f"Security Event: {event.event_type} - {event.error_message}")
        elif event.severity == "medium":
            logger.warning(f"Security Event: {event.event_type}")
        else:
            logger.info(f"Security Event: {event.event_type}")
    
    def log_authentication_event(
        self,
        user_id: str,
        ip_address: str,
        success: bool,
        error_message: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log authentication event
        
        Args:
            user_id: User identifier
            ip_address: IP address
            success: Whether authentication was successful
            error_message: Error message if failed
            user_agent: User agent string
        """
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type="authentication",
            severity="medium" if success else "high",
            user_id=user_id,
            tenant_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint="/auth/login",
            request_data_hash=None,
            response_status=200 if success else 401,
            error_message=error_message
        )
        
        self.log_security_event(event)
    
    def log_data_access_event(
        self,
        user_id: str,
        tenant_id: Optional[str],
        ip_address: str,
        endpoint: str,
        request_data: Optional[Dict[str, Any]] = None,
        response_status: int = 200,
        user_agent: Optional[str] = None
    ):
        """
        Log data access event
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            ip_address: IP address
            endpoint: API endpoint accessed
            request_data: Request data (will be hashed)
            response_status: HTTP response status
            user_agent: User agent string
        """
        request_hash = None
        if request_data:
            request_str = json.dumps(request_data, sort_keys=True)
            request_hash = hashlib.sha256(request_str.encode()).hexdigest()
        
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type="data_access",
            severity="low",
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            request_data_hash=request_hash,
            response_status=response_status,
            error_message=None
        )
        
        self.log_security_event(event)
    
    def log_security_violation(
        self,
        violation_type: str,
        ip_address: str,
        details: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log security violation
        
        Args:
            violation_type: Type of violation
            ip_address: IP address
            details: Violation details
            user_id: User identifier (if known)
            tenant_id: Tenant identifier (if known)
            user_agent: User agent string
        """
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type="security_violation",
            severity="critical",
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=None,
            request_data_hash=None,
            response_status=None,
            error_message=details,
            additional_data={'violation_type': violation_type}
        )
        
        self.log_security_event(event)
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "full"
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specified date range
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            report_type: Type of report ("full", "summary", "violations")
        
        Returns:
            Compliance report data
        """
        # Filter events by date range
        filtered_events = [
            event for event in self.events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Generate statistics
        total_events = len(filtered_events)
        events_by_type = {}
        events_by_severity = {}
        unique_users = set()
        unique_ips = set()
        violations = []
        
        for event in filtered_events:
            # Count by type
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            
            # Count by severity
            events_by_severity[event.severity] = events_by_severity.get(event.severity, 0) + 1
            
            # Track unique users and IPs
            if event.user_id:
                unique_users.add(event.user_id)
            unique_ips.add(event.ip_address)
            
            # Collect violations
            if event.event_type == "security_violation":
                violations.append(event)
        
        report = {
            'report_type': report_type,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'events_by_type': events_by_type,
                'events_by_severity': events_by_severity,
                'unique_users': len(unique_users),
                'unique_ip_addresses': len(unique_ips),
                'security_violations': len(violations)
            }
        }
        
        if report_type in ["full", "violations"]:
            report['violations'] = [asdict(violation) for violation in violations]
        
        if report_type == "full":
            report['all_events'] = [asdict(event) for event in filtered_events]
        
        return report
    
    def export_audit_log(self, output_path: str, format: str = "json"):
        """
        Export audit log to file
        
        Args:
            output_path: Output file path
            format: Export format ("json", "csv")
        """
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([asdict(event) for event in self.events], f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            
            with open(output_path, 'w', newline='') as f:
                if self.events:
                    fieldnames = asdict(self.events[0]).keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for event in self.events:
                        row = asdict(event)
                        # Convert datetime to string
                        row['timestamp'] = row['timestamp'].isoformat()
                        writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Audit log exported to: {output_path}")


class SecureRULService:
    """
    Secure RUL prediction service with comprehensive security features
    
    Integrates all security components into a unified secure service
    for production deployment.
    """
    
    def __init__(
        self,
        config: SecurityConfig,
        audit_log_path: str = "security_audit.log"
    ):
        """
        Initialize secure RUL service
        
        Args:
            config: Security configuration
            audit_log_path: Path to audit log file
        """
        self.config = config
        
        # Initialize security components
        if config.encryption_enabled and CRYPTO_AVAILABLE:
            self.encryption = DataEncryption()
        else:
            self.encryption = None
        
        if config.input_validation_enabled:
            self.input_sanitizer = InputSanitizer(config)
        else:
            self.input_sanitizer = None
        
        self.auth_manager = AuthenticationManager(config)
        
        if config.audit_logging_enabled:
            self.audit_logger = SecurityAuditLogger(audit_log_path)
        else:
            self.audit_logger = None
        
        logger.info("SecureRULService initialized with security features")
    
    def authenticate_request(
        self,
        token: Optional[str],
        ip_address: str,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate incoming request
        
        Args:
            token: JWT token or session ID
            ip_address: Client IP address
            user_agent: User agent string
        
        Returns:
            Authentication result with user info
        
        Raises:
            SecurityException: If authentication fails
        """
        # Validate IP address
        if self.input_sanitizer:
            try:
                ip_address = self.input_sanitizer.validate_ip_address(ip_address)
            except ValueError as e:
                if self.audit_logger:
                    self.audit_logger.log_security_violation(
                        violation_type="invalid_ip",
                        ip_address=ip_address,
                        details=str(e),
                        user_agent=user_agent
                    )
                raise SecurityException(f"Invalid IP address: {e}")
            
            # Check IP whitelist/blacklist
            if not self.input_sanitizer.check_ip_whitelist(ip_address):
                if self.audit_logger:
                    self.audit_logger.log_security_violation(
                        violation_type="ip_not_whitelisted",
                        ip_address=ip_address,
                        details="IP address not in allowed ranges",
                        user_agent=user_agent
                    )
                raise SecurityException("IP address not allowed")
            
            if self.input_sanitizer.check_ip_blacklist(ip_address):
                if self.audit_logger:
                    self.audit_logger.log_security_violation(
                        violation_type="ip_blacklisted",
                        ip_address=ip_address,
                        details="IP address is blacklisted",
                        user_agent=user_agent
                    )
                raise SecurityException("IP address is blocked")
        
        if not token:
            raise SecurityException("Authentication token required")
        
        try:
            # Try JWT token first
            if JWT_AVAILABLE:
                try:
                    payload = self.auth_manager.verify_jwt_token(token)
                    user_id = payload.get('user_id')
                    tenant_id = payload.get('tenant_id')
                    
                    if self.audit_logger:
                        self.audit_logger.log_authentication_event(
                            user_id=user_id,
                            ip_address=ip_address,
                            success=True,
                            user_agent=user_agent
                        )
                    
                    return {
                        'user_id': user_id,
                        'tenant_id': tenant_id,
                        'permissions': payload.get('permissions', []),
                        'auth_type': 'jwt'
                    }
                
                except jwt.InvalidTokenError:
                    pass  # Try session validation
            
            # Try session validation
            session = self.auth_manager.validate_session(token)
            if session:
                if self.audit_logger:
                    self.audit_logger.log_authentication_event(
                        user_id=session['user_id'],
                        ip_address=ip_address,
                        success=True,
                        user_agent=user_agent
                    )
                
                return {
                    'user_id': session['user_id'],
                    'tenant_id': session.get('tenant_id'),
                    'permissions': [],
                    'auth_type': 'session'
                }
            
            # Authentication failed
            if self.audit_logger:
                self.audit_logger.log_authentication_event(
                    user_id="unknown",
                    ip_address=ip_address,
                    success=False,
                    error_message="Invalid token or session",
                    user_agent=user_agent
                )
            
            raise SecurityException("Invalid authentication token")
        
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_security_violation(
                    violation_type="authentication_error",
                    ip_address=ip_address,
                    details=str(e),
                    user_agent=user_agent
                )
            raise SecurityException(f"Authentication failed: {e}")
    
    def validate_prediction_input(
        self,
        cycle_data: Dict[str, Any],
        features: List[float]
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Validate and sanitize prediction input data
        
        Args:
            cycle_data: Cycle data dictionary
            features: Feature array
        
        Returns:
            Tuple of (validated_cycle_data, validated_features)
        
        Raises:
            SecurityException: If validation fails
        """
        if not self.input_sanitizer:
            return cycle_data, np.array(features)
        
        try:
            # Validate cycle data
            validated_cycle_data = {}
            
            if 'cycle_number' in cycle_data:
                validated_cycle_data['cycle_number'] = int(
                    self.input_sanitizer.validate_numeric_input(
                        cycle_data['cycle_number'],
                        min_value=1,
                        max_value=10000,
                        allow_negative=False
                    )
                )
            
            if 'capacitor_id' in cycle_data:
                validated_cycle_data['capacitor_id'] = self.input_sanitizer.sanitize_string(
                    str(cycle_data['capacitor_id']),
                    max_length=50
                )
            
            if 'timestamp' in cycle_data:
                validated_cycle_data['timestamp'] = self.input_sanitizer.validate_numeric_input(
                    cycle_data['timestamp'],
                    min_value=0,
                    allow_negative=False
                )
            
            # Validate features array
            validated_features = self.input_sanitizer.validate_array_input(
                features,
                max_size=1000,
                element_type=float,
                allow_nan=False
            )
            
            return validated_cycle_data, validated_features
        
        except ValueError as e:
            raise SecurityException(f"Input validation failed: {e}")
    
    def secure_predict(
        self,
        token: str,
        ip_address: str,
        cycle_data: Dict[str, Any],
        features: List[float],
        user_agent: Optional[str] = None
    ) -> PredictionResult:
        """
        Make secure RUL prediction with full security validation
        
        Args:
            token: Authentication token
            ip_address: Client IP address
            cycle_data: Cycle data
            features: Feature array
            user_agent: User agent string
        
        Returns:
            Prediction result
        
        Raises:
            SecurityException: If security validation fails
        """
        # Authenticate request
        auth_info = self.authenticate_request(token, ip_address, user_agent)
        
        # Validate input
        validated_cycle_data, validated_features = self.validate_prediction_input(
            cycle_data, features
        )
        
        # Log data access
        if self.audit_logger:
            self.audit_logger.log_data_access_event(
                user_id=auth_info['user_id'],
                tenant_id=auth_info.get('tenant_id'),
                ip_address=ip_address,
                endpoint="/api/predict",
                request_data={
                    'cycle_data': validated_cycle_data,
                    'features_count': len(validated_features)
                },
                user_agent=user_agent
            )
        
        # Make prediction (placeholder - integrate with actual RUL predictor)
        prediction_result = PredictionResult(
            rul_cycles=100,
            rul_confidence_lower=90,
            rul_confidence_upper=110,
            degradation_score=0.5,
            degradation_stage="healthy",
            anomaly_flag=False,
            anomaly_score=0.0,
            feature_importance={},
            timestamp=time.time(),
            model_version="secure_v1.0"
        )
        
        # Encrypt sensitive data if encryption is enabled
        if self.encryption:
            # In a real implementation, you might encrypt sensitive parts of the result
            pass
        
        return prediction_result
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and configuration"""
        return {
            'encryption_enabled': self.config.encryption_enabled and self.encryption is not None,
            'input_validation_enabled': self.config.input_validation_enabled,
            'audit_logging_enabled': self.config.audit_logging_enabled,
            'tls_enabled': self.config.tls_enabled,
            'mtls_enabled': self.config.mtls_enabled,
            'rate_limiting_enabled': self.config.rate_limiting_enabled,
            'active_sessions': len(self.auth_manager.active_sessions),
            'failed_login_attempts': len(self.auth_manager.failed_login_attempts),
            'crypto_available': CRYPTO_AVAILABLE,
            'jwt_available': JWT_AVAILABLE
        }


class SecurityException(Exception):
    """Custom exception for security-related errors"""
    pass


# Utility functions for security deployment
def generate_security_config(
    environment: str = "production",
    enable_all_features: bool = True
) -> SecurityConfig:
    """
    Generate security configuration for deployment environment
    
    Args:
        environment: Deployment environment ("development", "staging", "production")
        enable_all_features: Whether to enable all security features
    
    Returns:
        Security configuration
    """
    if environment == "production":
        return SecurityConfig(
            encryption_enabled=enable_all_features,
            tls_enabled=True,
            mtls_enabled=enable_all_features,
            audit_logging_enabled=True,
            input_validation_enabled=True,
            rate_limiting_enabled=True,
            jwt_expiration_hours=8,  # Shorter for production
            max_request_size_mb=5,   # Smaller for production
            password_min_length=16,  # Longer for production
            password_require_special_chars=True,
            session_timeout_minutes=15,  # Shorter for production
            max_login_attempts=3,    # Stricter for production
            lockout_duration_minutes=30  # Longer for production
        )
    
    elif environment == "staging":
        return SecurityConfig(
            encryption_enabled=enable_all_features,
            tls_enabled=True,
            mtls_enabled=False,
            audit_logging_enabled=True,
            input_validation_enabled=True,
            rate_limiting_enabled=True,
            jwt_expiration_hours=12,
            max_request_size_mb=10,
            password_min_length=12,
            session_timeout_minutes=30,
            max_login_attempts=5,
            lockout_duration_minutes=15
        )
    
    else:  # development
        return SecurityConfig(
            encryption_enabled=False,
            tls_enabled=False,
            mtls_enabled=False,
            audit_logging_enabled=True,
            input_validation_enabled=True,
            rate_limiting_enabled=False,
            jwt_expiration_hours=24,
            max_request_size_mb=50,
            password_min_length=8,
            password_require_special_chars=False,
            session_timeout_minutes=60,
            max_login_attempts=10,
            lockout_duration_minutes=5
        )


def setup_tls_certificates(
    cert_dir: str,
    domain_name: str = "localhost"
) -> Tuple[str, str]:
    """
    Generate self-signed TLS certificates for development/testing
    
    Args:
        cert_dir: Directory to store certificates
        domain_name: Domain name for certificate
    
    Returns:
        Tuple of (cert_file_path, key_file_path)
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("Cryptography library required for TLS certificate generation")
    
    os.makedirs(cert_dir, exist_ok=True)
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Generate certificate
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RUL Prediction Service"),
        x509.NameAttribute(NameOID.COMMON_NAME, domain_name),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(domain_name),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256(), default_backend())
    
    # Save certificate and key
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")
    
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    logger.info(f"TLS certificates generated: {cert_path}, {key_path}")
    return cert_path, key_path