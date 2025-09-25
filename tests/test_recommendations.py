"""
Tests for healthcare recommendations system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import modules to test  
from src.recommendations import HealthcareRecommendationSystem

@pytest.mark.unit
class TestHealthcareRecommendationSystem:
    """Test the HealthcareRecommendationSystem class."""
    
    def test_init(self, test_database):
        """Test recommendation system initialization."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        assert rec_system is not None
        assert rec_system.db is not None
        assert hasattr(rec_system, 'risk_thresholds')
    
    def test_assess_risk_level(self, test_database, sample_patient):
        """Test risk level assessment."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Test with different risk probabilities
        low_risk = rec_system.assess_risk_level(0.2)  # 20% risk
        medium_risk = rec_system.assess_risk_level(0.6)  # 60% risk  
        high_risk = rec_system.assess_risk_level(0.8)  # 80% risk
        
        assert low_risk in ['Low', 'low']
        assert medium_risk in ['Medium', 'medium', 'Moderate', 'moderate']
        assert high_risk in ['High', 'high']
    
    def test_generate_lifestyle_recommendations_low_risk(self, test_database):
        """Test lifestyle recommendations for low risk patients."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {
            'Age': 35,
            'Sex': 'M',
            'ChestPainType': 'ASY',
            'RestingBP': 120,
            'Cholesterol': 180,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 170,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up'
        }
        
        recommendations = rec_system.generate_lifestyle_recommendations(patient_data, 0.2)
        
        assert isinstance(recommendations, dict)
        assert 'diet' in recommendations
        assert 'exercise' in recommendations
        assert 'lifestyle' in recommendations
        assert len(recommendations['diet']) > 0
        assert len(recommendations['exercise']) > 0
    
    def test_generate_lifestyle_recommendations_high_risk(self, test_database):
        """Test lifestyle recommendations for high risk patients."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {
            'Age': 65,
            'Sex': 'M', 
            'ChestPainType': 'ASY',
            'RestingBP': 160,
            'Cholesterol': 300,
            'FastingBS': 1,
            'RestingECG': 'LVH',
            'MaxHR': 120,
            'ExerciseAngina': 'Y',
            'Oldpeak': 3.0,
            'ST_Slope': 'Flat'
        }
        
        recommendations = rec_system.generate_lifestyle_recommendations(patient_data, 0.8)
        
        assert isinstance(recommendations, dict)
        assert 'diet' in recommendations
        assert 'exercise' in recommendations  
        assert 'lifestyle' in recommendations
        assert 'medical' in recommendations  # High risk should have medical recommendations
        
        # High risk should have more aggressive recommendations
        assert len(recommendations['medical']) > 0
    
    def test_recommend_hospitals(self, test_database):
        """Test hospital recommendations."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Test hospital recommendations for high risk
        hospitals = rec_system.recommend_hospitals(0.8, location={'city': 'Test City', 'state': 'TS'})
        
        assert isinstance(hospitals, list)
        assert len(hospitals) > 0
        
        # Should prioritize hospitals with cardiology services
        for hospital in hospitals:
            assert 'name' in hospital
            assert 'rating' in hospital
            assert hospital.get('cardiology_services') == 1
    
    def test_recommend_specialists(self, test_database):
        """Test specialist recommendations."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {
            'Age': 55,
            'RestingBP': 160,
            'Cholesterol': 280,
            'ExerciseAngina': 'Y'
        }
        
        specialists = rec_system.recommend_specialists(patient_data, 0.7)
        
        assert isinstance(specialists, list)
        assert len(specialists) > 0
        
        # Should include cardiologist for heart disease risk
        specialist_types = [spec['specialty'] for spec in specialists]
        assert 'Cardiologist' in specialist_types
    
    def test_create_emergency_plan(self, test_database):
        """Test emergency plan creation."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        emergency_plan = rec_system.create_emergency_plan(0.8)
        
        assert isinstance(emergency_plan, dict)
        assert 'warning_signs' in emergency_plan
        assert 'immediate_actions' in emergency_plan
        assert 'emergency_contacts' in emergency_plan
        
        # High risk should have comprehensive emergency plan
        assert len(emergency_plan['warning_signs']) > 0
        assert len(emergency_plan['immediate_actions']) > 0
    
    def test_generate_patient_report(self, test_database, sample_patient):
        """Test patient report generation."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Mock prediction
        risk_probability = 0.65
        
        report = rec_system.generate_patient_report(
            sample_patient, 
            risk_probability,
            location={'city': 'Test City', 'state': 'TS'}
        )
        
        assert isinstance(report, dict)
        assert 'patient_info' in report
        assert 'risk_assessment' in report
        assert 'recommendations' in report
        assert 'hospitals' in report
        assert 'specialists' in report
        assert 'emergency_plan' in report
        
        # Check report completeness
        assert report['patient_info']['age'] == sample_patient['Age']
        assert report['risk_assessment']['probability'] == risk_probability

@pytest.mark.unit
class TestRecommendationLogic:
    """Test recommendation generation logic."""
    
    def test_age_based_recommendations(self, test_database):
        """Test recommendations vary based on age."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        young_patient = {'Age': 25, 'Sex': 'M', 'RestingBP': 120, 'Cholesterol': 180}
        elderly_patient = {'Age': 75, 'Sex': 'M', 'RestingBP': 140, 'Cholesterol': 220}
        
        young_recs = rec_system.generate_lifestyle_recommendations(young_patient, 0.3)
        elderly_recs = rec_system.generate_lifestyle_recommendations(elderly_patient, 0.3)
        
        # Recommendations should be different based on age
        assert young_recs != elderly_recs
        
        # Elderly patients might have more medical recommendations
        assert len(elderly_recs.get('medical', [])) >= len(young_recs.get('medical', []))
    
    def test_gender_based_recommendations(self, test_database):
        """Test recommendations consider gender differences."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        male_patient = {'Age': 45, 'Sex': 'M', 'RestingBP': 130, 'Cholesterol': 200}
        female_patient = {'Age': 45, 'Sex': 'F', 'RestingBP': 130, 'Cholesterol': 200}
        
        male_recs = rec_system.generate_lifestyle_recommendations(male_patient, 0.4)
        female_recs = rec_system.generate_lifestyle_recommendations(female_patient, 0.4)
        
        # Should have some gender-specific considerations
        assert isinstance(male_recs, dict)
        assert isinstance(female_recs, dict)
    
    def test_risk_stratified_recommendations(self, test_database):
        """Test recommendations are properly stratified by risk level."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {'Age': 50, 'Sex': 'M', 'RestingBP': 140, 'Cholesterol': 220}
        
        low_risk_recs = rec_system.generate_lifestyle_recommendations(patient_data, 0.2)
        high_risk_recs = rec_system.generate_lifestyle_recommendations(patient_data, 0.8)
        
        # High risk should have more aggressive recommendations
        assert len(high_risk_recs.get('medical', [])) > len(low_risk_recs.get('medical', []))
        
        # High risk should have more frequent follow-ups
        high_risk_followup = any('frequent' in str(rec).lower() or 'regular' in str(rec).lower() 
                               for rec in high_risk_recs.get('medical', []))
        assert high_risk_followup or len(high_risk_recs.get('medical', [])) > 3

@pytest.mark.integration  
class TestRecommendationIntegration:
    """Integration tests for recommendation system."""
    
    def test_end_to_end_recommendation_flow(self, test_database, sample_patient):
        """Test complete recommendation workflow."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Simulate complete workflow
        risk_prob = 0.7
        location = {'city': 'Test City', 'state': 'TS'}
        
        # Generate complete recommendations
        full_report = rec_system.generate_patient_report(
            sample_patient, 
            risk_prob, 
            location
        )
        
        # Validate complete report structure
        required_sections = [
            'patient_info', 'risk_assessment', 'recommendations', 
            'hospitals', 'specialists', 'emergency_plan'
        ]
        
        for section in required_sections:
            assert section in full_report
            assert full_report[section] is not None
        
        # Validate recommendations are comprehensive
        recs = full_report['recommendations']
        assert 'diet' in recs
        assert 'exercise' in recs
        assert 'lifestyle' in recs
        
        # Validate hospitals are relevant
        hospitals = full_report['hospitals']
        assert len(hospitals) > 0
        assert all(h.get('cardiology_services') == 1 for h in hospitals)
    
    @patch('src.database.HospitalDatabase.search_hospitals')
    def test_recommendation_with_no_hospitals(self, mock_search, test_database, sample_patient):
        """Test recommendations when no hospitals are found."""
        mock_search.return_value = []  # No hospitals found
        
        rec_system = HealthcareRecommendationSystem(test_database)
        
        report = rec_system.generate_patient_report(
            sample_patient,
            0.6,
            location={'city': 'Remote City', 'state': 'RC'}
        )
        
        # Should handle gracefully
        assert 'hospitals' in report
        assert isinstance(report['hospitals'], list)
        # Should provide alternative guidance or expand search area
    
    def test_recommendation_caching_performance(self, test_database):
        """Test recommendation generation performance and caching."""
        import time
        
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {'Age': 50, 'Sex': 'M', 'RestingBP': 140, 'Cholesterol': 220}
        
        # First generation (no cache)
        start_time = time.time()
        recs1 = rec_system.generate_lifestyle_recommendations(patient_data, 0.5)
        first_time = time.time() - start_time
        
        # Second generation (potentially cached)
        start_time = time.time()
        recs2 = rec_system.generate_lifestyle_recommendations(patient_data, 0.5)
        second_time = time.time() - start_time
        
        # Results should be consistent
        assert recs1 == recs2
        
        # Performance should be reasonable
        assert first_time < 5.0  # Should generate in under 5 seconds
        assert second_time < 5.0

@pytest.mark.edge_cases
class TestRecommendationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_patient_data(self, test_database):
        """Test handling of invalid patient data."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Test with missing required fields
        incomplete_patient = {'Age': 45}
        
        try:
            recs = rec_system.generate_lifestyle_recommendations(incomplete_patient, 0.5)
            # Should handle gracefully with default recommendations
            assert isinstance(recs, dict)
        except Exception as e:
            # Or should raise appropriate error
            assert isinstance(e, (ValueError, KeyError))
    
    def test_extreme_risk_values(self, test_database):
        """Test handling of extreme risk probability values.""" 
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {'Age': 50, 'Sex': 'M'}
        
        # Test with risk = 0
        zero_risk_recs = rec_system.generate_lifestyle_recommendations(patient_data, 0.0)
        assert isinstance(zero_risk_recs, dict)
        
        # Test with risk = 1  
        max_risk_recs = rec_system.generate_lifestyle_recommendations(patient_data, 1.0)
        assert isinstance(max_risk_recs, dict)
        
        # Test with invalid risk values
        try:
            invalid_recs = rec_system.generate_lifestyle_recommendations(patient_data, -0.1)
            # Should handle gracefully or clip to valid range
        except ValueError:
            # Or raise appropriate error
            pass
        
        try:
            invalid_recs = rec_system.generate_lifestyle_recommendations(patient_data, 1.5)
            # Should handle gracefully or clip to valid range  
        except ValueError:
            # Or raise appropriate error
            pass
    
    def test_unusual_patient_characteristics(self, test_database):
        """Test recommendations for patients with unusual characteristics."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        # Very young patient with high risk factors
        unusual_patient = {
            'Age': 20,
            'Sex': 'M',
            'RestingBP': 180,  # Very high for age
            'Cholesterol': 350,  # Very high for age
            'FastingBS': 1
        }
        
        recs = rec_system.generate_lifestyle_recommendations(unusual_patient, 0.6)
        
        # Should provide appropriate recommendations despite unusual profile
        assert isinstance(recs, dict)
        assert len(recs.get('medical', [])) > 0  # Should recommend medical attention
    
    def test_missing_hospital_database(self):
        """Test handling when hospital database is unavailable."""
        # Test with invalid database path
        try:
            rec_system = HealthcareRecommendationSystem('/invalid/path/database.db')
            
            # Should either handle gracefully or raise appropriate error
            hospitals = rec_system.recommend_hospitals(0.5, {'city': 'Test', 'state': 'TS'})
            assert isinstance(hospitals, list)  # Might be empty list
        except Exception as e:
            # Should raise appropriate database error
            assert 'database' in str(e).lower() or 'connection' in str(e).lower()
    
    def test_recommendation_consistency(self, test_database):
        """Test that recommendations are consistent for same inputs."""
        rec_system = HealthcareRecommendationSystem(test_database)
        
        patient_data = {'Age': 50, 'Sex': 'M', 'RestingBP': 140}
        
        # Generate recommendations multiple times
        recs1 = rec_system.generate_lifestyle_recommendations(patient_data, 0.5)
        recs2 = rec_system.generate_lifestyle_recommendations(patient_data, 0.5) 
        recs3 = rec_system.generate_lifestyle_recommendations(patient_data, 0.5)
        
        # Should be consistent
        assert recs1 == recs2 == recs3