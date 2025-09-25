"""
Tests for database operations and hospital management.
"""

import pytest
import sqlite3
import pandas as pd
import os
from unittest.mock import patch, Mock

# Import modules to test
from src.database import (
    create_synthetic_hospital_data,
    create_sqlite_database,
    query_hospitals_by_location
)

@pytest.mark.database
class TestHospitalDatabase:
    """Test the HospitalDatabase class."""
    
    def test_init_with_existing_db(self, test_database):
        """Test initialization with existing database."""
        db = HospitalDatabase(test_database)
        assert db.db_path == test_database
        assert db.connection is not None
    
    def test_init_creates_new_db(self, test_model_dir):
        """Test initialization creates new database if not exists."""
        new_db_path = os.path.join(test_model_dir, 'new_test.db')
        
        db = HospitalDatabase(new_db_path)
        
        # Should create database and tables
        assert os.path.exists(new_db_path)
        assert db.connection is not None
        
        # Should have hospitals table
        cursor = db.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hospitals'")
        assert cursor.fetchone() is not None
    
    def test_add_hospital(self, test_database):
        """Test adding a new hospital."""
        db = HospitalDatabase(test_database)
        
        hospital_data = {
            'name': 'New Test Hospital',
            'address': '789 New St',
            'city': 'New City',
            'state': 'NS',
            'zip_code': '54321',
            'phone': '555-9999',
            'website': 'www.newtest.com',
            'cardiology_services': 1,
            'emergency_services': 1,
            'rating': 4.3,
            'specialties': 'Cardiology,Surgery'
        }
        
        result = db.add_hospital(**hospital_data)
        
        assert result is True
        
        # Verify hospital was added
        hospitals = db.get_all_hospitals()
        assert len(hospitals) == 3  # 2 initial + 1 new
        
        # Check if new hospital exists
        new_hospital = [h for h in hospitals if h['name'] == 'New Test Hospital']
        assert len(new_hospital) == 1
        assert new_hospital[0]['city'] == 'New City'
    
    def test_get_all_hospitals(self, test_database):
        """Test retrieving all hospitals."""
        db = HospitalDatabase(test_database)
        hospitals = db.get_all_hospitals()
        
        assert len(hospitals) == 2  # From test fixture
        assert all('name' in hospital for hospital in hospitals)
        assert all('city' in hospital for hospital in hospitals)
        assert all('rating' in hospital for hospital in hospitals)
    
    def test_search_hospitals_by_location(self, test_database):
        """Test searching hospitals by location."""
        db = HospitalDatabase(test_database)
        
        # Search by city
        results = db.search_hospitals(city='Test City')
        assert len(results) == 2
        
        # Search by state
        results = db.search_hospitals(state='TS')
        assert len(results) == 2
        
        # Search by nonexistent location
        results = db.search_hospitals(city='Nonexistent City')
        assert len(results) == 0
    
    def test_search_hospitals_by_services(self, test_database):
        """Test searching hospitals by services."""
        db = HospitalDatabase(test_database)
        
        # Search hospitals with cardiology services
        results = db.search_hospitals(cardiology_services=True)
        assert len(results) == 2
        
        # Search hospitals with emergency services
        results = db.search_hospitals(emergency_services=True)
        assert len(results) == 2
    
    def test_search_hospitals_by_rating(self, test_database):
        """Test searching hospitals by minimum rating."""
        db = HospitalDatabase(test_database)
        
        # Search hospitals with rating >= 4.3
        results = db.search_hospitals(min_rating=4.3)
        assert len(results) == 1  # Only Test Heart Hospital has 4.5 rating
        
        # Search hospitals with rating >= 4.0
        results = db.search_hospitals(min_rating=4.0)
        assert len(results) == 2  # Both hospitals have rating >= 4.0
    
    def test_get_nearby_hospitals(self, test_database):
        """Test getting nearby hospitals (mock geolocation)."""
        db = HospitalDatabase(test_database)
        
        with patch.object(db, '_calculate_distance') as mock_distance:
            mock_distance.return_value = 5.0  # Mock 5 km distance
            
            # This would normally use geolocation APIs
            nearby = db.get_nearby_hospitals('Test City', 'TS', radius_km=10)
            
            assert len(nearby) == 2  # Both hospitals in Test City
    
    def test_get_hospital_by_id(self, test_database):
        """Test retrieving hospital by ID."""
        db = HospitalDatabase(test_database)
        
        hospital = db.get_hospital_by_id(1)
        assert hospital is not None
        assert hospital['name'] == 'Test Heart Hospital'
        assert hospital['id'] == 1
        
        # Test nonexistent ID
        hospital = db.get_hospital_by_id(999)
        assert hospital is None
    
    def test_update_hospital(self, test_database):
        """Test updating hospital information."""
        db = HospitalDatabase(test_database)
        
        # Update hospital rating
        result = db.update_hospital(1, rating=4.8, phone='555-UPDATED')
        assert result is True
        
        # Verify update
        hospital = db.get_hospital_by_id(1)
        assert hospital['rating'] == 4.8
        assert hospital['phone'] == '555-UPDATED'
    
    def test_delete_hospital(self, test_database):
        """Test deleting a hospital."""
        db = HospitalDatabase(test_database)
        
        # Delete hospital
        result = db.delete_hospital(2)
        assert result is True
        
        # Verify deletion
        hospitals = db.get_all_hospitals()
        assert len(hospitals) == 1
        
        hospital = db.get_hospital_by_id(2)
        assert hospital is None

@pytest.mark.database
class TestDatabaseConnectivity:
    """Test database connection and error handling."""
    
    def test_connection_handling(self, test_database):
        """Test database connection management."""
        db = HospitalDatabase(test_database)
        
        # Should have active connection
        assert db.connection is not None
        
        # Test connection is working
        cursor = db.connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
    
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Try to connect to invalid database path
        invalid_path = '/invalid/path/database.db'
        
        # Should handle gracefully or raise appropriate error
        try:
            db = HospitalDatabase(invalid_path)
            # If successful, should still be functional
            assert db.connection is not None
        except Exception as e:
            # Should raise appropriate database error
            assert isinstance(e, (sqlite3.Error, OSError, IOError))
    
    def test_concurrent_access(self, test_database):
        """Test concurrent database access."""
        db1 = HospitalDatabase(test_database)
        db2 = HospitalDatabase(test_database)
        
        # Both should be able to read
        hospitals1 = db1.get_all_hospitals()
        hospitals2 = db2.get_all_hospitals()
        
        assert len(hospitals1) == len(hospitals2)
        assert hospitals1[0]['name'] == hospitals2[0]['name']

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_bulk_hospital_operations(self, test_database):
        """Test bulk operations on hospitals."""
        db = HospitalDatabase(test_database)
        
        # Add multiple hospitals
        hospitals_to_add = [
            {
                'name': f'Hospital {i}',
                'address': f'{i}00 Medical St',
                'city': 'Bulk City',
                'state': 'BC',
                'zip_code': f'1000{i}',
                'phone': f'555-000{i}',
                'website': f'www.hospital{i}.com',
                'cardiology_services': 1,
                'emergency_services': 1,
                'rating': 4.0 + (i * 0.1),
                'specialties': 'Cardiology'
            }
            for i in range(1, 6)
        ]
        
        # Add all hospitals
        for hospital in hospitals_to_add:
            result = db.add_hospital(**hospital)
            assert result is True
        
        # Verify all were added
        all_hospitals = db.get_all_hospitals()
        assert len(all_hospitals) == 7  # 2 original + 5 new
        
        # Search by bulk city
        bulk_hospitals = db.search_hospitals(city='Bulk City')
        assert len(bulk_hospitals) == 5
    
    def test_database_backup_restore(self, test_database, test_model_dir):
        """Test database backup and restore operations."""
        db = HospitalDatabase(test_database)
        
        # Add a new hospital
        db.add_hospital(
            name='Backup Test Hospital',
            address='123 Backup St',
            city='Backup City',
            state='BC',
            zip_code='99999',
            phone='555-BACKUP',
            website='www.backup.com'
        )
        
        # Create backup by copying database file
        backup_path = os.path.join(test_model_dir, 'backup.db')
        
        # Simple backup (copy database file)
        import shutil
        shutil.copy2(test_database, backup_path)
        
        # Verify backup works
        backup_db = HospitalDatabase(backup_path)
        backup_hospitals = backup_db.get_all_hospitals()
        
        # Should have same number of hospitals including the new one
        original_hospitals = db.get_all_hospitals()
        assert len(backup_hospitals) == len(original_hospitals)
        
        # Find the backup test hospital
        backup_hospital = [h for h in backup_hospitals if h['name'] == 'Backup Test Hospital']
        assert len(backup_hospital) == 1

@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    def test_large_dataset_query_performance(self, test_database):
        """Test query performance with larger dataset."""
        import time
        
        db = HospitalDatabase(test_database)
        
        # Add many hospitals for performance testing
        start_time = time.time()
        
        for i in range(100):
            db.add_hospital(
                name=f'Perf Hospital {i}',
                address=f'{i} Performance St',
                city=f'City {i % 10}',  # 10 different cities
                state='PF',
                zip_code=f'{10000 + i}',
                phone=f'555-{1000 + i}',
                website=f'www.perf{i}.com',
                rating=4.0 + ((i % 10) * 0.1)
            )
        
        add_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        results = db.search_hospitals(state='PF', min_rating=4.5)
        query_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert add_time < 10.0  # Should add 100 records in under 10 seconds
        assert query_time < 1.0  # Should query in under 1 second
        assert len(results) > 0  # Should find some results

@pytest.mark.edge_cases  
class TestDatabaseEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_hospital_data(self, test_database):
        """Test handling of invalid hospital data."""
        db = HospitalDatabase(test_database)
        
        # Test missing required fields
        with pytest.raises((ValueError, sqlite3.Error, TypeError)):
            db.add_hospital(name='Test')  # Missing required fields
        
        # Test invalid data types
        with pytest.raises((ValueError, sqlite3.Error, TypeError)):
            db.add_hospital(
                name=123,  # Should be string
                address='123 Test St',
                city='Test City',
                state='TS'
            )
    
    def test_duplicate_hospital_handling(self, test_database):
        """Test handling of duplicate hospital data."""
        db = HospitalDatabase(test_database)
        
        # Add hospital twice with same details
        hospital_data = {
            'name': 'Duplicate Hospital',
            'address': '123 Duplicate St',
            'city': 'Duplicate City',
            'state': 'DC',
            'phone': '555-DUPE'
        }
        
        # First addition should succeed
        result1 = db.add_hospital(**hospital_data)
        assert result1 is True
        
        # Second addition might succeed (creating duplicate) or fail
        # Depending on database constraints
        try:
            result2 = db.add_hospital(**hospital_data)
            # If it succeeds, we now have a duplicate
            hospitals = db.search_hospitals(name='Duplicate Hospital')
            assert len(hospitals) >= 1
        except Exception:
            # If it fails due to constraints, that's also valid
            pass
    
    def test_sql_injection_protection(self, test_database):
        """Test protection against SQL injection."""
        db = HospitalDatabase(test_database)
        
        # Try SQL injection in search
        malicious_city = "'; DROP TABLE hospitals; --"
        
        # Should not cause database corruption
        try:
            results = db.search_hospitals(city=malicious_city)
            # Should return no results, not cause error
            assert len(results) == 0
        except Exception:
            # Or should raise appropriate error, not crash
            pass
        
        # Verify database is still intact
        hospitals = db.get_all_hospitals()
        assert len(hospitals) >= 2  # Should still have original test data
    
    def test_very_long_field_values(self, test_database):
        """Test handling of very long field values."""
        db = HospitalDatabase(test_database)
        
        # Create very long strings
        very_long_name = 'A' * 1000
        very_long_address = 'B' * 2000
        
        try:
            result = db.add_hospital(
                name=very_long_name,
                address=very_long_address,
                city='Test City',
                state='TS'
            )
            
            # If successful, should truncate or handle appropriately
            if result:
                hospitals = db.search_hospitals(city='Test City')
                long_hospital = [h for h in hospitals if h['name'].startswith('AAA')]
                if long_hospital:
                    # Name might be truncated
                    assert len(long_hospital[0]['name']) <= 1000
        except Exception:
            # Or should raise appropriate error for field length
            pass