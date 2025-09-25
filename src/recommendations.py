"""
Healthcare recommendations system for heart failure prediction.
Provides personalized lifestyle advice and hospital suggestions based on predictions.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from typing import Dict, List, Tuple, Optional

class HealthcareRecommendationSystem:
    """
    Comprehensive healthcare recommendation system that provides:
    1. Risk-based lifestyle recommendations
    2. Hospital and specialist suggestions
    3. Preventive care guidance
    4. Emergency action plans
    """
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_hospitals.db')
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.7,
            'high': 1.0
        }
        
    def assess_risk_level(self, prediction_probability: float) -> str:
        """Categorize risk level based on prediction probability."""
        if prediction_probability < self.risk_thresholds['low']:
            return 'low'
        elif prediction_probability < self.risk_thresholds['moderate']:
            return 'moderate'
        else:
            return 'high'
    
    def get_lifestyle_recommendations(self, patient_data: Dict, prediction_probability: float) -> Dict:
        """
        Generate personalized lifestyle recommendations based on patient data and risk level.
        """
        risk_level = self.assess_risk_level(prediction_probability)
        recommendations = {
            'risk_level': risk_level,
            'probability': prediction_probability,
            'diet': [],
            'exercise': [],
            'lifestyle': [],
            'monitoring': [],
            'emergency_signs': []
        }
        
        # Risk-based general recommendations
        if risk_level == 'high':
            recommendations['diet'].extend([
                "Follow a strict low-sodium diet (less than 2,000mg per day)",
                "Adopt a DASH diet rich in fruits, vegetables, and whole grains",
                "Limit saturated fats to less than 7% of total calories",
                "Avoid processed foods and excessive alcohol consumption"
            ])
            
            recommendations['exercise'].extend([
                "Consult cardiologist before starting any exercise program",
                "Begin with 10-15 minutes of light walking daily",
                "Gradually increase to 150 minutes of moderate exercise per week",
                "Include strength training 2 days per week (with medical approval)"
            ])
            
            recommendations['lifestyle'].extend([
                "Quit smoking immediately - seek professional help",
                "Manage stress through meditation, yoga, or counseling",
                "Ensure 7-9 hours of quality sleep daily",
                "Take prescribed medications consistently"
            ])
            
            recommendations['monitoring'].extend([
                "Monitor blood pressure daily",
                "Check weight weekly and report sudden changes",
                "Regular cholesterol and glucose monitoring",
                "Schedule cardiology appointments every 3 months"
            ])
            
        elif risk_level == 'moderate':
            recommendations['diet'].extend([
                "Reduce sodium intake to less than 2,300mg per day",
                "Increase intake of omega-3 fatty acids (fish, nuts)",
                "Choose lean proteins and limit red meat",
                "Include 5+ servings of fruits and vegetables daily"
            ])
            
            recommendations['exercise'].extend([
                "Aim for 150 minutes of moderate aerobic activity weekly",
                "Include 2-3 days of strength training",
                "Try activities like brisk walking, swimming, or cycling",
                "Monitor heart rate during exercise"
            ])
            
            recommendations['lifestyle'].extend([
                "If you smoke, create a quit plan",
                "Practice stress management techniques",
                "Maintain a healthy weight (BMI 18.5-24.9)",
                "Limit alcohol to 1-2 drinks per day"
            ])
            
            recommendations['monitoring'].extend([
                "Check blood pressure regularly",
                "Annual cholesterol screening",
                "Regular check-ups with primary care physician",
                "Track symptoms and report changes"
            ])
            
        else:  # low risk
            recommendations['diet'].extend([
                "Maintain a balanced, heart-healthy diet",
                "Keep sodium intake reasonable (less than 2,300mg/day)",
                "Focus on whole foods over processed options",
                "Stay hydrated and limit sugary beverages"
            ])
            
            recommendations['exercise'].extend([
                "Continue regular physical activity",
                "Aim for at least 150 minutes moderate exercise weekly",
                "Include variety: cardio, strength, and flexibility",
                "Try new activities to maintain motivation"
            ])
            
            recommendations['lifestyle'].extend([
                "Continue healthy habits",
                "Don't smoke or quit if you do",
                "Manage stress effectively",
                "Maintain healthy relationships and social connections"
            ])
            
            recommendations['monitoring'].extend([
                "Annual health check-ups",
                "Monitor blood pressure occasionally",
                "Stay aware of family history",
                "Report any new or concerning symptoms"
            ])
        
        # Feature-specific recommendations
        if 'Cholesterol' in patient_data:
            cholesterol = patient_data['Cholesterol']
            if cholesterol > 200:  # Assuming normalized values
                recommendations['diet'].append("Focus on cholesterol-lowering foods (oats, beans, nuts)")
                recommendations['monitoring'].append("Monitor cholesterol levels every 6 months")
        
        if 'RestingBP' in patient_data:
            bp = patient_data['RestingBP']
            if bp > 140:  # Assuming actual BP values
                recommendations['diet'].append("Follow a low-sodium, DASH diet strictly")
                recommendations['monitoring'].append("Monitor blood pressure daily")
        
        if 'ExerciseAngina' in patient_data and patient_data['ExerciseAngina'] == 1:
            recommendations['exercise'] = ["Consult cardiologist before any exercise", "Start with very light activity"]
            recommendations['monitoring'].append("Report chest pain immediately")
        
        # Emergency warning signs (always included)
        recommendations['emergency_signs'] = [
            "Chest pain or discomfort lasting more than few minutes",
            "Shortness of breath with or without chest pain",
            "Pain in arms, neck, jaw, shoulder, or upper back",
            "Nausea, vomiting, lightheadedness, or breaking out in cold sweat",
            "Extreme fatigue or weakness",
            "Rapid or irregular heartbeat"
        ]
        
        return recommendations
    
    def find_hospitals(self, city: Optional[str] = None, state: Optional[str] = None, 
                      emergency: bool = False, max_results: int = 10) -> List[Dict]:
        """
        Find hospitals based on location and requirements.
        """
        if not os.path.exists(self.db_path):
            return [{"error": "Hospital database not found"}]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query based on parameters - use OR for city/state to be more flexible
            query = "SELECT * FROM hospitals WHERE 1=1"
            params = []
            
            if emergency:
                query += " AND emergency_services = 1"
            
            # Use OR condition for city and state to be more flexible with synthetic data
            location_conditions = []
            if city:
                location_conditions.append("LOWER(city) = LOWER(?)")
                params.append(city)
            
            if state:
                location_conditions.append("UPPER(state) = UPPER(?)")
                params.append(state)
            
            if location_conditions:
                query += " AND (" + " OR ".join(location_conditions) + ")"
            
            query += " ORDER BY cardiac_rating DESC"
            
            if max_results:
                query += f" LIMIT {max_results}"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            columns = ['id', 'name', 'address', 'city', 'state', 'zip_code', 'phone', 
                      'cardiac_rating', 'emergency_services', 'specializations', 'beds_count', 'website']
            
            hospitals = []
            for row in results:
                hospital = dict(zip(columns, row))
                hospital['emergency_services'] = bool(hospital['emergency_services'])
                hospitals.append(hospital)
            
            return hospitals
            
        except Exception as e:
            return [{"error": f"Database query failed: {e}"}]
    
    def get_specialist_recommendations(self, risk_level: str, patient_data: Dict) -> Dict:
        """
        Recommend appropriate specialists based on risk level and patient characteristics.
        """
        specialists = {
            'primary': [],
            'secondary': [],
            'emergency': []
        }
        
        # Primary care (always recommended)
        specialists['primary'].append({
            'type': 'Primary Care Physician',
            'reason': 'Regular monitoring and coordination of care',
            'frequency': 'Every 3-6 months' if risk_level == 'high' else 'Annually'
        })
        
        # Risk-based specialist recommendations
        if risk_level in ['high', 'moderate']:
            specialists['primary'].append({
                'type': 'Cardiologist',
                'reason': 'Specialized heart care and treatment planning',
                'frequency': 'Every 3 months' if risk_level == 'high' else 'Every 6 months'
            })
            
            if risk_level == 'high':
                specialists['secondary'].extend([
                    {
                        'type': 'Electrophysiologist',
                        'reason': 'If experiencing irregular heart rhythms',
                        'frequency': 'As needed'
                    },
                    {
                        'type': 'Heart Failure Specialist',
                        'reason': 'Specialized management of heart failure symptoms',
                        'frequency': 'As recommended by cardiologist'
                    }
                ])
        
        # Feature-specific specialists
        if 'Cholesterol' in patient_data and patient_data['Cholesterol'] > 200:
            specialists['secondary'].append({
                'type': 'Lipid Specialist',
                'reason': 'Management of high cholesterol levels',
                'frequency': 'Every 6 months'
            })
        
        if 'RestingBP' in patient_data and patient_data['RestingBP'] > 140:
            specialists['secondary'].append({
                'type': 'Hypertension Specialist',
                'reason': 'Management of high blood pressure',
                'frequency': 'Every 3 months'
            })
        
        # Support specialists
        specialists['secondary'].extend([
            {
                'type': 'Dietitian/Nutritionist',
                'reason': 'Personalized dietary planning',
                'frequency': 'Initial consultation, then as needed'
            },
            {
                'type': 'Exercise Physiologist',
                'reason': 'Safe exercise program development',
                'frequency': 'Initial consultation and periodic updates'
            }
        ])
        
        if risk_level == 'high':
            specialists['secondary'].append({
                'type': 'Mental Health Counselor',
                'reason': 'Stress management and coping strategies',
                'frequency': 'As needed'
            })
        
        return specialists
    
    def create_action_plan(self, patient_data: Dict, prediction_probability: float,
                          city: Optional[str] = None, state: Optional[str] = None) -> Dict:
        """
        Create a comprehensive action plan for the patient.
        """
        risk_level = self.assess_risk_level(prediction_probability)
        
        action_plan = {
            'patient_summary': {
                'risk_level': risk_level,
                'probability': prediction_probability,
                'assessment_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            },
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'lifestyle_recommendations': self.get_lifestyle_recommendations(patient_data, prediction_probability),
            'specialist_recommendations': self.get_specialist_recommendations(risk_level, patient_data),
            'hospital_suggestions': self.find_hospitals(city, state, emergency=(risk_level=='high')),
            'emergency_plan': self.get_emergency_plan(risk_level)
        }
        
        # Immediate actions based on risk level
        if risk_level == 'high':
            action_plan['immediate_actions'].extend([
                "Schedule urgent cardiology consultation within 1-2 weeks",
                "Begin prescribed medications if any",
                "Avoid strenuous activities until cleared by doctor",
                "Monitor symptoms closely and record them"
            ])
        elif risk_level == 'moderate':
            action_plan['immediate_actions'].extend([
                "Schedule cardiology consultation within 1 month",
                "Begin heart-healthy lifestyle changes immediately",
                "Monitor blood pressure and symptoms",
                "Review medications with healthcare provider"
            ])
        else:
            action_plan['immediate_actions'].extend([
                "Continue current healthy lifestyle",
                "Schedule routine check-up with primary care",
                "Stay aware of risk factors and family history"
            ])
        
        # Short-term actions (1-3 months)
        action_plan['short_term_actions'].extend([
            "Implement dietary changes gradually",
            "Establish regular exercise routine",
            "Complete baseline health screenings",
            "Build support network for lifestyle changes"
        ])
        
        # Long-term actions (3+ months)
        action_plan['long_term_actions'].extend([
            "Maintain lifestyle modifications",
            "Regular follow-ups with healthcare team",
            "Monitor progress and adjust plan as needed",
            "Stay informed about heart health research"
        ])
        
        return action_plan
    
    def get_emergency_plan(self, risk_level: str) -> Dict:
        """
        Create an emergency action plan based on risk level.
        """
        emergency_plan = {
            'when_to_call_911': [
                "Chest pain lasting more than 5 minutes",
                "Severe shortness of breath",
                "Loss of consciousness",
                "Severe nausea with chest discomfort",
                "Any symptoms that feel life-threatening"
            ],
            'emergency_contacts': [
                "Emergency Services: 911",
                "Poison Control: 1-800-222-1222",
                "Primary Care Doctor: [Add your doctor's number]",
                "Cardiologist: [Add cardiologist's number if applicable]"
            ],
            'emergency_kit': [
                "List of current medications",
                "Emergency contact information",
                "Insurance information",
                "Medical history summary",
                "Aspirin (if recommended by doctor)"
            ]
        }
        
        if risk_level == 'high':
            emergency_plan['additional_precautions'] = [
                "Always carry emergency contact information",
                "Inform family/friends about your condition",
                "Consider medical alert bracelet",
                "Keep nitroglycerin if prescribed",
                "Have someone available to drive you to hospital"
            ]
        
        return emergency_plan

def generate_patient_report(patient_data: Dict, prediction_probability: float,
                           city: Optional[str] = None, state: Optional[str] = None) -> str:
    """
    Generate a comprehensive patient report with recommendations.
    """
    system = HealthcareRecommendationSystem()
    action_plan = system.create_action_plan(patient_data, prediction_probability, city, state)
    
    report = f"""
# Heart Health Assessment Report

## Assessment Summary
- **Risk Level**: {action_plan['patient_summary']['risk_level'].upper()}
- **Prediction Probability**: {action_plan['patient_summary']['probability']:.2%}
- **Assessment Date**: {action_plan['patient_summary']['assessment_date']}

## Immediate Actions Required
"""
    
    for action in action_plan['immediate_actions']:
        report += f"- {action}\n"
    
    report += "\n## Lifestyle Recommendations\n\n"
    
    # Diet recommendations
    report += "### Dietary Guidelines\n"
    for diet_rec in action_plan['lifestyle_recommendations']['diet']:
        report += f"- {diet_rec}\n"
    
    # Exercise recommendations
    report += "\n### Exercise Recommendations\n"
    for exercise_rec in action_plan['lifestyle_recommendations']['exercise']:
        report += f"- {exercise_rec}\n"
    
    # Monitoring recommendations
    report += "\n### Monitoring Requirements\n"
    for monitor_rec in action_plan['lifestyle_recommendations']['monitoring']:
        report += f"- {monitor_rec}\n"
    
    # Emergency warning signs
    report += "\n## Emergency Warning Signs\n"
    report += "**Call 911 immediately if you experience:**\n"
    for sign in action_plan['lifestyle_recommendations']['emergency_signs']:
        report += f"- {sign}\n"
    
    # Hospital recommendations
    if action_plan['hospital_suggestions'] and 'error' not in action_plan['hospital_suggestions'][0]:
        report += "\n## Recommended Healthcare Facilities\n"
        for i, hospital in enumerate(action_plan['hospital_suggestions'][:3], 1):
            report += f"\n### {i}. {hospital['name']}\n"
            report += f"- **Address**: {hospital['address']}, {hospital['city']}, {hospital['state']}\n"
            report += f"- **Phone**: {hospital['phone']}\n"
            report += f"- **Cardiac Rating**: {hospital['cardiac_rating']}/5.0\n"
            report += f"- **Emergency Services**: {'Yes' if hospital['emergency_services'] else 'No'}\n"
            report += f"- **Specializations**: {hospital['specializations']}\n"
    
    report += "\n## Disclaimer\n"
    report += "This assessment is for informational purposes only and should not replace professional medical advice. "
    report += "Please consult with qualified healthcare providers for proper diagnosis and treatment.\n"
    
    return report

def main():
    """Demo of the healthcare recommendation system."""
    print("=== Healthcare Recommendation System Demo ===\n")
    
    # Sample patient data
    sample_patient = {
        'Age': 55,
        'Sex': 1,  # Male
        'Cholesterol': 250,
        'RestingBP': 145,
        'ExerciseAngina': 0,
        'MaxHR': 120
    }
    
    # Sample prediction probability (high risk)
    prediction_prob = 0.85
    
    # Initialize system
    system = HealthcareRecommendationSystem()
    
    # Generate recommendations
    print("Generating comprehensive health recommendations...")
    action_plan = system.create_action_plan(sample_patient, prediction_prob, city="New York", state="NY")
    
    print(f"Risk Level: {action_plan['patient_summary']['risk_level'].upper()}")
    print(f"Prediction Probability: {prediction_prob:.2%}")
    
    print("\nLifestyle Recommendations:")
    for category, recommendations in action_plan['lifestyle_recommendations'].items():
        if category not in ['risk_level', 'probability'] and recommendations:
            print(f"  {category.title()}:")
            for rec in recommendations[:2]:  # Show first 2 recommendations
                print(f"    - {rec}")
    
    print(f"\nFound {len(action_plan['hospital_suggestions'])} hospitals in the area")
    if action_plan['hospital_suggestions'] and 'error' not in action_plan['hospital_suggestions'][0]:
        top_hospital = action_plan['hospital_suggestions'][0]
        print(f"Top recommended: {top_hospital['name']} (Rating: {top_hospital['cardiac_rating']}/5.0)")
    
    # Generate full report
    report = generate_patient_report(sample_patient, prediction_prob, "New York", "NY")
    
    # Save report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'sample_patient_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nFull patient report saved to: {report_path}")
    print("\n=== Recommendation System Ready ===")
    
    return system

if __name__ == "__main__":
    recommendation_system = main()