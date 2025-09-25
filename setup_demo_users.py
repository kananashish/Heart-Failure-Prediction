"""
Demo setup script for Heart Failure Prediction System.
Creates sample users for testing the authentication system.
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.auth import AuthenticationSystem

def create_demo_users():
    """Create demo users for testing."""
    print("🚀 Setting up demo users for Heart Failure Prediction System...")
    
    auth_system = AuthenticationSystem()
    
    # Demo users to create
    demo_users = [
        {
            'username': 'admin',
            'email': 'admin@heartprediction.com',
            'password': 'Admin123!',
            'full_name': 'System Administrator'
        },
        {
            'username': 'doctor_smith',
            'email': 'doctor.smith@hospital.com',
            'password': 'Doctor123!',
            'full_name': 'Dr. Jane Smith'
        },
        {
            'username': 'demo_user',
            'email': 'demo@example.com',
            'password': 'Demo123!',
            'full_name': 'Demo User'
        }
    ]
    
    created_count = 0
    for user_data in demo_users:
        success, message = auth_system.register_user(
            username=user_data['username'],
            email=user_data['email'],
            password=user_data['password'],
            full_name=user_data['full_name']
        )
        
        if success:
            print(f"✅ Created user: {user_data['username']} ({user_data['full_name']})")
            created_count += 1
        else:
            print(f"❌ Failed to create user {user_data['username']}: {message}")
    
    print(f"\n🎉 Demo setup complete! Created {created_count} users.")
    print("\n📋 Demo Login Credentials:")
    print("=" * 50)
    for user_data in demo_users:
        print(f"Username: {user_data['username']}")
        print(f"Password: {user_data['password']}")
        print(f"Email: {user_data['email']}")
        print("-" * 30)
    
    print("\n🌐 Access the application at: http://localhost:8501")
    print("💡 Tip: Try logging in with any of the demo accounts above!")

if __name__ == "__main__":
    create_demo_users()