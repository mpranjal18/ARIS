"""Pest and disease risk computation helpers for explainable output."""

from __future__ import annotations

import pandas as pd


def compute_pest_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    """Returns pest component columns for explainability."""
    out = df.copy()
    out["aphid_component"] = out["aphid_population"]
    out["rust_component"] = out["rust_severity"]
    out["armyworm_component"] = out["armyworm_presence"]
    return out
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add config and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import config
from utils.data_handler import DataHandler

class PestDiseaseRiskAssessor:
    """
    Class for assessing pest and disease risks for wheat cultivation
    """
    
    def __init__(self):
        self.handler = DataHandler()
        self.risk_levels = config.RISK_LEVELS
        self.pest_factors = config.PEST_DISEASE_FACTORS
        
    def assess_aphid_risk(self, pest_data):
        """
        Assess risk based on aphid population levels
        """
        # Aphid risk is directly proportional to population (0-1 scale)
        pest_data['aphid_risk'] = pest_data['aphid_population']
        return pest_data
    
    def assess_rust_risk(self, pest_data, weather_data=None):
        """
        Assess rust disease risk based on pest data and optionally weather conditions
        Rust favors high humidity and moderate temperatures
        """
        # Base rust risk from severity data
        pest_data['rust_risk'] = pest_data['rust_severity']
        
        # If weather data is provided, enhance risk assessment
        if weather_data is not None:
            # Merge with weather data
            merged_data = pd.merge(pest_data, weather_data, on=['date', 'region'], how='left')
            
            # High humidity increases rust risk
            humidity_factor = np.clip(merged_data['humidity'] / 100, 0, 1)
            
            # Optimal temperature for rust is 15-25°C
            temp_factor = np.where(
                (merged_data['temp_min'] >= 10) & (merged_data['temp_max'] <= 30),
                1.0,  # Favorable temperature
                0.5   # Less favorable
            )
            
            # Enhanced rust risk considering weather
            enhanced_rust_risk = merged_data['rust_severity'] * humidity_factor * temp_factor
            pest_data['rust_risk'] = np.clip(enhanced_rust_risk, 0, 1)
            
        return pest_data
    
    def assess_armyworm_risk(self, pest_data):
        """
        Assess armyworm risk based on presence data
        """
        # Armyworm risk is directly proportional to presence (0-1 scale)
        pest_data['armyworm_risk'] = pest_data['armyworm_presence']
        return pest_data
    
    def compute_pest_disease_risk_score(self, pest_data):
        """
        Compute overall pest and disease risk score based on all factors
        """
        # Initialize risk components
        risk_components = {}
        
        # Aphid risk
        if 'aphid_risk' in pest_data.columns:
            risk_components['aphid_population'] = pest_data['aphid_risk']
        else:
            risk_components['aphid_population'] = np.zeros(len(pest_data))
        
        # Rust risk
        if 'rust_risk' in pest_data.columns:
            risk_components['rust_severity'] = pest_data['rust_risk']
        else:
            risk_components['rust_severity'] = np.zeros(len(pest_data))
        
        # Armyworm risk
        if 'armyworm_risk' in pest_data.columns:
            risk_components['armyworm_presence'] = pest_data['armyworm_risk']
        else:
            risk_components['armyworm_presence'] = np.zeros(len(pest_data))
        
        # Weighted combination of all factors
        # Equal weights for now, but can be adjusted based on domain knowledge
        weights = {factor: 1/len(config.PEST_DISEASE_FACTORS) for factor in config.PEST_DISEASE_FACTORS}
        
        # Calculate weighted risk score
        weighted_scores = np.zeros(len(pest_data))
        for factor, weight in weights.items():
            weighted_scores += risk_components[factor] * weight
        
        # Normalize to 0-1 range
        pest_disease_risk_score = np.clip(weighted_scores, 0, 1)
        
        # Convert to categorical risk level
        def score_to_level(score):
            if score < 0.33:
                return 'Low'
            elif score < 0.66:
                return 'Medium'
            else:
                return 'High'
        
        pest_data['pest_disease_risk_score'] = pest_disease_risk_score
        pest_data['pest_disease_risk_level'] = pest_data['pest_disease_risk_score'].apply(score_to_level)
        
        return pest_data
    
    def assess_pest_disease_risk(self, pest_data, weather_data=None):
        """
        Main method to assess pest and disease risk
        """
        # Process all pest and disease factors
        pest_data = self.assess_aphid_risk(pest_data)
        pest_data = self.assess_rust_risk(pest_data, weather_data)
        pest_data = self.assess_armyworm_risk(pest_data)
        
        # Compute overall pest and disease risk
        pest_data = self.compute_pest_disease_risk_score(pest_data)
        
        return pest_data

# Example usage
if __name__ == "__main__":
    # Initialize assessor
    assessor = PestDiseaseRiskAssessor()
    
    # Generate sample data
    handler = DataHandler()
    pest_data = handler.generate_synthetic_pest_data()
    weather_data = handler.generate_synthetic_weather_data()
    
    # Assess pest and disease risk
    risk_data = assessor.assess_pest_disease_risk(pest_data, weather_data)
    
    # Display results
    print("Pest & Disease Risk Assessment Results:")
    print(risk_data[['date', 'region', 'pest_disease_risk_score', 'pest_disease_risk_level']].head(10))