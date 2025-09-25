"""
Hospital database creation and management module.
Creates a comprehensive database of hospitals with cardiology services.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import random
from typing import List, Dict

def create_synthetic_hospital_data(n_hospitals=200):
    """
    Create a synthetic hospital dataset with cardiology services.
    In a real scenario, you would download from healthcare.gov or similar sources.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Common hospital names and suffixes
    hospital_names = [
        "General Hospital", "Medical Center", "Heart Institute", "Regional Medical Center",
        "University Hospital", "Community Hospital", "Memorial Hospital", "Baptist Hospital",
        "Methodist Hospital", "Presbyterian Hospital", "St. Mary's Hospital", "Sacred Heart Hospital",
        "Children's Hospital", "Veterans Hospital", "City Hospital", "County Medical Center",
        "Cardiac Care Center", "Heart & Vascular Institute", "Cardiovascular Hospital"
    ]
    
    city_prefixes = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
        "San Francisco", "Columbus", "Indianapolis", "Fort Worth", "Charlotte", "Seattle",
        "Denver", "Washington", "Boston", "Nashville", "Baltimore", "Portland", "Las Vegas",
        "Oklahoma City", "Detroit", "Memphis", "Louisville", "Milwaukee", "Albuquerque",
        "Tucson", "Fresno", "Sacramento", "Mesa", "Kansas City", "Atlanta", "Omaha",
        "Colorado Springs", "Raleigh", "Miami", "Virginia Beach", "Oakland", "Minneapolis",
        "Tampa", "Arlington", "Wichita", "Bakersfield", "New Orleans", "Honolulu", "Anaheim"
    ]
    
    states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
        "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
        "VA", "WA", "WV", "WI", "WY"
    ]
    
    # Generate hospital data
    hospitals = []
    for i in range(n_hospitals):
        city = random.choice(city_prefixes)
        state = random.choice(states)
        hospital_type = random.choice(hospital_names)
        
        # Create hospital name
        if random.random() < 0.3:
            name = f"{city} {hospital_type}"
        else:
            name = f"{random.choice(['St. Joseph', 'Mount Sinai', 'Cedar Sinai', 'Mayo Clinic', 'Cleveland Clinic', 'Johns Hopkins', city])} {hospital_type}"
        
        # Create address
        street_number = random.randint(100, 9999)
        street_names = ["Main St", "Hospital Dr", "Medical Ave", "Health Blvd", "Care Way", "Healing Rd", "University Ave"]
        address = f"{street_number} {random.choice(street_names)}"
        
        # Generate quality scores and specializations
        cardiac_rating = round(random.uniform(3.5, 5.0), 1)
        emergency_services = random.choices([True, False], weights=[85, 15], k=1)[0]  # 85% have emergency services
        
        # Specializations
        specializations = []
        cardiac_specialties = [
            "Interventional Cardiology", "Electrophysiology", "Heart Surgery", 
            "Cardiac Rehabilitation", "Preventive Cardiology", "Heart Failure",
            "Congenital Heart Disease", "Vascular Surgery"
        ]
        
        # Each hospital has 2-5 cardiac specializations
        num_specializations = random.randint(2, 5)
        specializations = random.sample(cardiac_specialties, num_specializations)
        
        hospital = {
            "Name": name,
            "Address": address,
            "City": city,
            "State": state,
            "ZipCode": f"{random.randint(10000, 99999)}",
            "Phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
            "CardiacRating": cardiac_rating,
            "EmergencyServices": emergency_services,
            "Specializations": ", ".join(specializations),
            "BedsCount": random.randint(50, 800),
            "Website": f"www.{name.lower().replace(' ', '').replace('.', '').replace("'", '')}.com"
        }
        hospitals.append(hospital)
    
    return pd.DataFrame(hospitals)

def create_sqlite_database(df: pd.DataFrame, db_path: str):
    """
    Create SQLite database from hospital DataFrame.
    """
    try:
        # Remove existing database if it exists
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # Create connection
        conn = sqlite3.connect(db_path)
        
        # Create hospitals table with proper schema
        create_table_sql = """
        CREATE TABLE hospitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            zip_code TEXT,
            phone TEXT,
            cardiac_rating REAL,
            emergency_services BOOLEAN,
            specializations TEXT,
            beds_count INTEGER,
            website TEXT
        );
        """
        
        conn.execute(create_table_sql)
        
        # Insert data
        for _, row in df.iterrows():
            insert_sql = """
            INSERT INTO hospitals (name, address, city, state, zip_code, phone, 
                                 cardiac_rating, emergency_services, specializations, beds_count, website)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn.execute(insert_sql, (
                row['Name'], row['Address'], row['City'], row['State'], row['ZipCode'],
                row['Phone'], row['CardiacRating'], row['EmergencyServices'],
                row['Specializations'], row['BedsCount'], row['Website']
            ))
        
        # Create indexes for better query performance
        conn.execute("CREATE INDEX idx_city ON hospitals(city);")
        conn.execute("CREATE INDEX idx_state ON hospitals(state);")
        conn.execute("CREATE INDEX idx_cardiac_rating ON hospitals(cardiac_rating);")
        
        conn.commit()
        print(f"Successfully created SQLite database with {len(df)} hospitals")
        
        # Test queries
        print("\n=== Database Query Test ===")
        cursor = conn.cursor()
        
        # Test query - top rated hospitals
        cursor.execute("SELECT name, city, state, cardiac_rating FROM hospitals WHERE cardiac_rating >= 4.5 ORDER BY cardiac_rating DESC LIMIT 5")
        top_hospitals = cursor.fetchall()
        print("Top 5 rated cardiac hospitals:")
        for hospital in top_hospitals:
            print(f"  - {hospital[0]} ({hospital[1]}, {hospital[2]}) - Rating: {hospital[3]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def query_hospitals_by_location(db_path: str, city: str = None, state: str = None):
    """
    Query hospitals by location.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if city and state:
        query = "SELECT * FROM hospitals WHERE LOWER(city) = LOWER(?) AND UPPER(state) = UPPER(?) ORDER BY cardiac_rating DESC"
        cursor.execute(query, (city, state))
    elif state:
        query = "SELECT * FROM hospitals WHERE UPPER(state) = UPPER(?) ORDER BY cardiac_rating DESC"
        cursor.execute(query, (state,))
    elif city:
        query = "SELECT * FROM hospitals WHERE LOWER(city) = LOWER(?) ORDER BY cardiac_rating DESC"
        cursor.execute(query, (city,))
    else:
        query = "SELECT * FROM hospitals ORDER BY cardiac_rating DESC LIMIT 10"
        cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    columns = ['id', 'name', 'address', 'city', 'state', 'zip_code', 'phone', 
               'cardiac_rating', 'emergency_services', 'specializations', 'beds_count', 'website']
    
    hospitals = []
    for row in results:
        hospital = dict(zip(columns, row))
        hospitals.append(hospital)
    
    return hospitals

def main():
    """
    Main function to create hospital database.
    """
    print("=== Hospital Database Creation ===\n")
    
    # Create synthetic hospital data
    print("Creating synthetic hospital dataset...")
    hospitals_df = create_synthetic_hospital_data(200)
    
    # Save as CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hospitals.csv')
    hospitals_df.to_csv(csv_path, index=False)
    print(f"Hospital CSV saved to {csv_path}")
    
    # Create SQLite database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_hospitals.db')
    print(f"\nCreating SQLite database at {db_path}...")
    success = create_sqlite_database(hospitals_df, db_path)
    
    if success:
        # Test location-based queries
        print("\n=== Testing Location-Based Queries ===")
        
        # Test New York hospitals
        ny_hospitals = query_hospitals_by_location(db_path, state="NY")
        print(f"Found {len(ny_hospitals)} hospitals in New York state")
        
        # Test Los Angeles hospitals
        la_hospitals = query_hospitals_by_location(db_path, city="Los Angeles", state="CA")
        print(f"Found {len(la_hospitals)} hospitals in Los Angeles, CA")
        
        # Print statistics
        print(f"\n=== Database Statistics ===")
        print(f"Total hospitals: {len(hospitals_df)}")
        print(f"Average cardiac rating: {hospitals_df['CardiacRating'].mean():.2f}")
        print(f"Hospitals with emergency services: {hospitals_df['EmergencyServices'].sum()}")
        print(f"States covered: {hospitals_df['State'].nunique()}")
        print(f"Cities covered: {hospitals_df['City'].nunique()}")
    
    return hospitals_df

if __name__ == "__main__":
    hospital_data = main()