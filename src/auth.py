"""
Authentication module for Heart Failure Prediction System.
Handles user registration, login, password hashing, and session management.
"""

import sqlite3
import hashlib
import secrets
import streamlit as st
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Tuple
import re

class AuthenticationSystem:
    def __init__(self, db_path: str = None):
        """Initialize authentication system with database."""
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.db')
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize user database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create sessions table for session management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, hash_value = stored_hash.split(':')
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == hash_value
        except:
            return False
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        return True, "Password is valid"
    
    def register_user(self, username: str, email: str, password: str, full_name: str) -> Tuple[bool, str]:
        """Register a new user."""
        # Validate inputs
        if not username or not email or not password or not full_name:
            return False, "All fields are required"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username or email already exists
            cursor.execute('SELECT username, email FROM users WHERE username = ? OR email = ?', 
                         (username, email))
            existing = cursor.fetchone()
            
            if existing:
                if existing[0] == username:
                    return False, "Username already exists"
                else:
                    return False, "Email already registered"
            
            # Hash password and create user
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            conn.commit()
            conn.close()
            
            return True, "User registered successfully"
        
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Authenticate user login."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find user by username or email
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if not user:
                return False, None
            
            # Verify password
            if not self.verify_password(password, user[3]):
                return False, None
            
            # Update last login
            cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user[0],))
            conn.commit()
            conn.close()
            
            # Return user info
            user_info = {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[4],
                'role': user[5]
            }
            
            return True, user_info
        
        except Exception as e:
            return False, None
    
    def create_session(self, user_id: int) -> str:
        """Create a new session for user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)  # 24 hour session
            
            # Deactivate old sessions for this user
            cursor.execute('UPDATE sessions SET is_active = 0 WHERE user_id = ?', (user_id,))
            
            # Create new session
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            conn.close()
            
            return session_token
        
        except Exception as e:
            return None
    
    def validate_session(self, session_token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate session token and return user info."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.user_id, s.expires_at, u.username, u.email, u.full_name, u.role
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.is_active = 1 AND u.is_active = 1
            ''', (session_token,))
            
            session = cursor.fetchone()
            conn.close()
            
            if not session:
                return False, None
            
            # Check if session expired
            expires_at = datetime.fromisoformat(session[1])
            if datetime.now() > expires_at:
                self.invalidate_session(session_token)
                return False, None
            
            user_info = {
                'id': session[0],
                'username': session[2],
                'email': session[3],
                'full_name': session[4],
                'role': session[5]
            }
            
            return True, user_info
        
        except Exception as e:
            return False, None
    
    def invalidate_session(self, session_token: str):
        """Invalidate a session token."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
        except:
            pass
    
    def logout_user(self, session_token: str):
        """Logout user by invalidating session."""
        self.invalidate_session(session_token)

def init_session_state():
    """Initialize session state for authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None

def show_login_form(auth_system: AuthenticationSystem):
    """Display login form."""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="auth-form">
            <div class="form-title">
                🔐 <span class="medical-icon">🩺</span> Healthcare Professional Login
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### 🏥 Secure Access Portal")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("👤", help="Username or Email")
            with col2:
                username = st.text_input("Username or Email", placeholder="Enter your medical ID or email", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🔒", help="Password")
            with col2:
                password = st.text_input("Password", type="password", placeholder="Enter your secure password", label_visibility="collapsed")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Access Medical System", type="primary")
            with col2:
                register_button = st.form_submit_button("📝 New Healthcare Provider?")
        
        if register_button:
            st.session_state.show_register = True
            st.rerun()
        
        if login_button:
            if username and password:
                with st.spinner("🔐 Verifying medical credentials..."):
                    success, user_info = auth_system.authenticate_user(username, password)
                    if success:
                        # Create session
                        session_token = auth_system.create_session(user_info['id'])
                        if session_token:
                            st.session_state.authenticated = True
                            st.session_state.user_info = user_info
                            st.session_state.session_token = session_token
                            st.success(f"🏥 Welcome back, Dr. {user_info['full_name']}!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Failed to create secure session. Please try again.")
                    else:
                        st.error("❌ Invalid medical credentials. Please check your username and password.")
            else:
                st.error("⚠️ Please enter both username and password to access the medical system.")
    
    # Additional information section
    st.markdown("""
    <div class="info-section">
        <h4 style="color: #e74c3c; margin-bottom: 1rem;">🏥 Healthcare Professional Access</h4>
        <p><strong>✓</strong> Secure patient data analysis</p>
        <p><strong>✓</strong> HIPAA-compliant predictions</p>
        <p><strong>✓</strong> Evidence-based risk assessment</p>
        <p><strong>✓</strong> Professional medical reporting</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_register_form(auth_system: AuthenticationSystem):
    """Display registration form."""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="auth-form">
            <div class="form-title">
                📝 <span class="medical-icon">👩‍⚕️</span> Healthcare Professional Registration
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form", clear_on_submit=False):
            st.markdown("### 🏥 Join Our Medical Network")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("👨‍⚕️", help="Full Name")
            with col2:
                full_name = st.text_input("Full Name", placeholder="Dr. Jane Smith", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🆔", help="Medical ID")
            with col2:
                username = st.text_input("Medical ID / Username", placeholder="medical_id_2024", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("📧", help="Professional Email")
            with col2:
                email = st.text_input("Professional Email", placeholder="doctor@hospital.com", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🔐", help="Secure Password")
            with col2:
                password = st.text_input("Secure Password", type="password", placeholder="Create a strong password", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🔒", help="Confirm Password")
            with col2:
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", label_visibility="collapsed")
            
            # Password requirements with medical theme
            st.markdown("""
            <div class="password-requirements">
                <h4 style="color: #e74c3c; margin-bottom: 0.8rem;">🔐 Medical-Grade Security Requirements</h4>
                <ul>
                    <li>🔹 Minimum 8 characters for enhanced protection</li>
                    <li>🔹 Mix of uppercase and lowercase letters</li>
                    <li>🔹 At least one numeric digit</li>
                    <li>🔹 Professional-grade password strength</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                register_button = st.form_submit_button("🎯 Register for Medical Access", type="primary")
            with col2:
                back_button = st.form_submit_button("🔙 Back to Login")
        
        if back_button:
            st.session_state.show_register = False
            st.rerun()
        
        if register_button:
            if not all([full_name, username, email, password, confirm_password]):
                st.error("⚠️ All medical credentials are required for registration.")
            elif password != confirm_password:
                st.error("🔒 Password confirmation doesn't match. Please verify your entries.")
            else:
                with st.spinner("🏥 Creating your medical professional account..."):
                    success, message = auth_system.register_user(username, email, password, full_name)
                    if success:
                        st.success("🎉 Medical professional account created successfully!")
                        st.success("🔐 Please login with your new credentials to access the system.")
                        st.balloons()
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error(f"❌ Registration failed: {message}")
    
    # Medical professional benefits section
    st.markdown("""
    <div class="benefits-section">
        <h4>👩‍⚕️ Healthcare Professional Benefits</h4>
        <div class="benefits-grid">
            <div>
                <p><strong>🔬</strong> Advanced ML predictions</p>
                <p><strong>📊</strong> Comprehensive analytics</p>
                <p><strong>📈</strong> Risk assessment tools</p>
            </div>
            <div>
                <p><strong>🏥</strong> Hospital recommendations</p>
                <p><strong>📋</strong> Professional reports</p>
                <p><strong>🔐</strong> Secure data handling</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_authentication_page():
    """Show authentication page with login/register forms."""
    # Custom CSS for authentication matching main app theme with theme support
    st.markdown("""
    <style>
        /* Theme-aware CSS variables */
        :root {
            --auth-bg-light: #ffffff;
            --auth-bg-dark: #262730;
            --text-light: #2c3e50;
            --text-dark: #ffffff;
            --secondary-bg-light: #f8f9fa;
            --secondary-bg-dark: #1e1e2e;
            --border-light: #e9ecef;
            --border-dark: #4a4b57;
        }
        
        /* Main authentication styling */
        .auth-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .auth-header {
            text-align: center;
            color: #e74c3c;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .welcome-banner {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #a93226 100%);
            color: white !important;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 10px 25px rgba(231, 76, 60, 0.3);
        }
        
        .welcome-title, .welcome-subtitle, .welcome-desc {
            color: white !important;
        }
        
        .welcome-title {
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .welcome-subtitle {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
            opacity: 0.95;
        }
        
        .welcome-desc {
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 1rem;
        }
        
        /* Theme-aware form styling */
        .auth-form {
            background: var(--auth-bg-light);
            border: 1px solid var(--border-light);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            margin: 1rem 0;
            position: relative;
        }
        
        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            .auth-form {
                background: var(--auth-bg-dark) !important;
                border: 1px solid var(--border-dark) !important;
                color: var(--text-dark) !important;
            }
            
            .form-title {
                color: var(--text-dark) !important;
            }
            
            .info-section, .benefits-section {
                background: var(--secondary-bg-dark) !important;
                color: var(--text-dark) !important;
            }
            
            .info-section p, .benefits-section p {
                color: #b0b0b0 !important;
            }
            
            .password-requirements {
                background: var(--secondary-bg-dark) !important;
                color: var(--text-dark) !important;
            }
            
            .password-requirements li {
                color: #b0b0b0 !important;
            }
        }
        
        .form-title {
            color: var(--text-light);
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
            position: relative;
        }
        
        .form-title::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 50px;
            height: 3px;
            background: #e74c3c;
            border-radius: 2px;
        }
        
        /* Info sections */
        .info-section {
            margin-top: 2rem;
            padding: 1.5rem;
            background: var(--secondary-bg-light);
            border-radius: 12px;
            border-left: 4px solid #e74c3c;
        }
        
        .info-section h4 {
            color: #e74c3c !important;
            margin-bottom: 1rem;
        }
        
        .info-section p {
            margin: 0.5rem 0;
            color: var(--text-light);
        }
        
        .benefits-section {
            margin-top: 2rem;
            padding: 2rem;
            background: var(--secondary-bg-light);
            border-radius: 12px;
            border-left: 4px solid #28a745;
        }
        
        .benefits-section h4 {
            color: #28a745 !important;
            margin-bottom: 1rem;
        }
        
        .benefits-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .benefits-section p {
            margin: 0.3rem 0;
            color: var(--text-light);
        }
        
        .password-requirements {
            background: var(--secondary-bg-light);
            border-left: 4px solid #17a2b8;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        
        .password-requirements h4 {
            color: #e74c3c !important;
            margin-bottom: 0.8rem;
        }
        
        .password-requirements li {
            margin: 0.3rem 0;
            color: var(--text-light);
        }
        
        /* Button styling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3) !important;
        }
        
        .stButton > button:not([kind="primary"]) {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .benefits-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Medical-themed welcome banner
    st.markdown("""
    <div class="welcome-banner">
        <div class="welcome-title">
            🫀 Heart Failure Prediction System
        </div>
        <div class="welcome-subtitle">
            Advanced Machine Learning Healthcare Solution
        </div>
        <div class="welcome-desc">
            Secure access required • Professional medical analysis • Evidence-based predictions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize auth system
    auth_system = AuthenticationSystem()
    
    # Initialize session state
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    # Show appropriate form
    if st.session_state.show_register:
        show_register_form(auth_system)
    else:
        show_login_form(auth_system)

def check_authentication():
    """Check if user is authenticated and validate session."""
    init_session_state()
    
    # If user claims to be authenticated, validate session
    if st.session_state.authenticated and st.session_state.session_token:
        auth_system = AuthenticationSystem()
        valid, user_info = auth_system.validate_session(st.session_state.session_token)
        
        if valid:
            st.session_state.user_info = user_info
            return True
        else:
            # Session invalid, reset state
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.session_token = None
    
    return st.session_state.authenticated

def logout_user():
    """Logout current user."""
    if st.session_state.session_token:
        auth_system = AuthenticationSystem()
        auth_system.logout_user(st.session_state.session_token)
    
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.session_token = None
    st.rerun()

def get_current_user() -> Optional[Dict]:
    """Get current authenticated user info."""
    return st.session_state.user_info if st.session_state.authenticated else None