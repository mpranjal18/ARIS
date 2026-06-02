"""Climate risk computation helpers used by analysis and reporting layers."""

from __future__ import annotations

import pandas as pd


def compute_climate_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    """Returns component-wise climate risk factors for explainability."""
    out = df.copy()
    out["rainfall_component"] = out["rainfall_deviation"].abs()
    out["temperature_component"] = out["temperature_anomaly"].abs()
    out["heatwave_component"] = out["heatwave_flag"]
    out["humidity_component"] = out["humidity_stress"]
    out["soil_moisture_component"] = out["soil_moisture_stress"]
    out["ndvi_component"] = 1 - out["ndvi"]
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

class ClimateRiskAssessor:
    """
    Class for assessing climate risks for wheat cultivation
    """
    
    def __init__(self):
        self.handler = DataHandler()
        self.risk_levels = config.RISK_LEVELS
        self.climate_factors = config.CLIMATE_FACTORS
        
    def calculate_rainfall_deviation(self, rainfall_data, window=30):
        """
        Calculate rainfall deviation from historical average
        """
        # Calculate rolling average
        rainfall_data['rainfall_avg'] = rainfall_data['rainfall'].rolling(window=window, min_periods=1).mean()
        rainfall_data['rainfall_deviation'] = rainfall_data['rainfall'] - rainfall_data['rainfall_avg']
        
        # Normalize deviation (-1 to 1)
        max_deviation = max(abs(rainfall_data['rainfall_deviation'].min()), 
                           abs(rainfall_data['rainfall_deviation'].max()))
        if max_deviation > 0:
            rainfall_data['rainfall_deviation_normalized'] = rainfall_data['rainfall_deviation'] / max_deviation
        else:
            rainfall_data['rainfall_deviation_normalized'] = 0
            
        return rainfall_data
    
    def detect_heatwaves(self, weather_data, temp_threshold=35, consecutive_days=3):
        """
        Detect heatwave events based on temperature threshold
        """
        # Identify hot days
        weather_data['is_hot_day'] = weather_data['temp_max'] >= temp_threshold
        
        # Calculate consecutive hot days
        weather_data['consecutive_hot_days'] = (
            weather_data.groupby((weather_data['is_hot_day'] != weather_data['is_hot_day'].shift()).cumsum())
            ['is_hot_day'].cumsum()
        )
        
        # Identify heatwave events
        weather_data['is_heatwave'] = weather_data['consecutive_hot_days'] >= consecutive_days
        
        # Heat stress score (0-1)
        weather_data['heatwave_score'] = weather_data['is_heatwave'].astype(int)
        
        return weather_data
    
    def detect_temperature_anomalies(self, weather_data, window=30):
        """
        Detect cold wave and heat anomalies
        """
        # Calculate rolling averages
        weather_data['temp_min_avg'] = weather_data['temp_min'].rolling(window=window, min_periods=1).mean()
        weather_data['temp_max_avg'] = weather_data['temp_max'].rolling(window=window, min_periods=1).mean()
        
        # Calculate deviations
        weather_data['temp_min_deviation'] = weather_data['temp_min'] - weather_data['temp_min_avg']
        weather_data['temp_max_deviation'] = weather_data['temp_max'] - weather_data['temp_max_avg']
        
        # Cold wave detection (significant drop in minimum temperature)
        cold_wave_threshold = np.percentile(weather_data['temp_min_deviation'].dropna(), 10)
        weather_data['is_cold_wave'] = weather_data['temp_min_deviation'] <= cold_wave_threshold
        
        # Heat anomaly detection (significant rise in maximum temperature)
        heat_anomaly_threshold = np.percentile(weather_data['temp_max_deviation'].dropna(), 90)
        weather_data['is_heat_anomaly'] = weather_data['temp_max_deviation'] >= heat_anomaly_threshold
        
        # Temperature anomaly score (0-1)
        weather_data['temperature_anomaly_score'] = (
            (weather_data['is_cold_wave'] | weather_data['is_heat_anomaly']).astype(int)
        )
        
        return weather_data
    
    def assess_humidity_risk(self, weather_data, optimal_range=(40, 70)):
        """
        Assess risk based on humidity levels (favors rust disease)
        """
        # Calculate how far humidity deviates from optimal range
        weather_data['humidity_risk'] = 0.0
        
        # High humidity risk (> 70%)
        high_humidity_mask = weather_data['humidity'] > optimal_range[1]
        weather_data.loc[high_humidity_mask, 'humidity_risk'] = (
            (weather_data.loc[high_humidity_mask, 'humidity'] - optimal_range[1]) / (100 - optimal_range[1])
        )
        
        # Low humidity risk (< 40%)
        low_humidity_mask = weather_data['humidity'] < optimal_range[0]
        weather_data.loc[low_humidity_mask, 'humidity_risk'] = (
            (optimal_range[0] - weather_data.loc[low_humidity_mask, 'humidity']) / optimal_range[0]
        )
        
        return weather_data
    
    def calculate_soil_moisture_index(self, ndvi_data):
        """
        Calculate soil moisture using LSWI (Land Surface Water Index)
        LSWI = (NIR - SWIR) / (NIR + SWIR)
        For simplicity, we'll derive it from NDVI and LSWI data
        """
        # In our synthetic data, LSWI is already provided
        # Calculate soil moisture stress index (0-1)
        ndvi_data['soil_moisture_stress'] = np.maximum(0, ndvi_data['ndvi'] - ndvi_data['lswi'])
        
        # Normalize to 0-1 range
        max_stress = ndvi_data['soil_moisture_stress'].max()
        if max_stress > 0:
            ndvi_data['soil_moisture_index'] = ndvi_data['soil_moisture_stress'] / max_stress
        else:
            ndvi_data['soil_moisture_index'] = 0
            
        return ndvi_data
    
    def analyze_ndvi_trend(self, ndvi_data, window=30):
        """
        Analyze NDVI trend to assess vegetation health
        """
        # Calculate rolling average
        ndvi_data['ndvi_smooth'] = ndvi_data['ndvi'].rolling(window=window, min_periods=1).mean()
        
        # Calculate trend using linear regression slope
        def calculate_trend(group):
            if len(group) < 2:
                return 0
            x = np.arange(len(group))
            y = group['ndvi_smooth'].values
            slope = np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0
            return slope
        
        ndvi_data['ndvi_trend'] = ndvi_data.groupby('region').apply(
            lambda x: pd.Series([calculate_trend(x)] * len(x), index=x.index)
        ).values.flatten()
        
        # Convert trend to health score (negative trend = poor health)
        ndvi_data['ndvi_health_score'] = np.clip(-ndvi_data['ndvi_trend'], 0, 1)
        
        return ndvi_data
    
    def compute_climate_risk_score(self, combined_data):
        """
        Compute overall climate risk score based on all factors
        """
        # Initialize risk components
        risk_components = {}
        
        # Rainfall deviation risk (normalized to 0-1)
        if 'rainfall_deviation_normalized' in combined_data.columns:
            risk_components['rainfall_deviation'] = np.abs(combined_data['rainfall_deviation_normalized'])
        else:
            risk_components['rainfall_deviation'] = np.zeros(len(combined_data))
        
        # Heatwave risk
        if 'heatwave_score' in combined_data.columns:
            risk_components['heatwaves'] = combined_data['heatwave_score']
        else:
            risk_components['heatwaves'] = np.zeros(len(combined_data))
        
        # Temperature anomaly risk
        if 'temperature_anomaly_score' in combined_data.columns:
            risk_components['temperature_anomalies'] = combined_data['temperature_anomaly_score']
        else:
            risk_components['temperature_anomalies'] = np.zeros(len(combined_data))
        
        # Humidity risk
        if 'humidity_risk' in combined_data.columns:
            risk_components['humidity'] = combined_data['humidity_risk']
        else:
            risk_components['humidity'] = np.zeros(len(combined_data))
        
        # Soil moisture risk
        if 'soil_moisture_index' in combined_data.columns:
            risk_components['soil_moisture'] = combined_data['soil_moisture_index']
        else:
            risk_components['soil_moisture'] = np.zeros(len(combined_data))
        
        # NDVI health risk
        if 'ndvi_health_score' in combined_data.columns:
            risk_components['ndvi_trend'] = combined_data['ndvi_health_score']
        else:
            risk_components['ndvi_trend'] = np.zeros(len(combined_data))
        
        # Weighted combination of all factors
        # Equal weights for now, but can be adjusted based on domain knowledge
        weights = {factor: 1/len(config.CLIMATE_FACTORS) for factor in config.CLIMATE_FACTORS}
        
        # Calculate weighted risk score
        weighted_scores = np.zeros(len(combined_data))
        for factor, weight in weights.items():
            weighted_scores += risk_components[factor] * weight
        
        # Normalize to 0-1 range
        climate_risk_score = np.clip(weighted_scores, 0, 1)
        
        # Convert to categorical risk level
        def score_to_level(score):
            if score < 0.33:
                return 'Low'
            elif score < 0.66:
                return 'Medium'
            else:
                return 'High'
        
        combined_data['climate_risk_score'] = climate_risk_score
        combined_data['climate_risk_level'] = combined_data['climate_risk_score'].apply(score_to_level)
        
        return combined_data
    
    def assess_climate_risk(self, weather_data, ndvi_data):
        """
        Main method to assess climate risk
        """
        # Process all climate factors
        weather_data = self.calculate_rainfall_deviation(weather_data)
        weather_data = self.detect_heatwaves(weather_data)
        weather_data = self.detect_temperature_anomalies(weather_data)
        weather_data = self.assess_humidity_risk(weather_data)
        
        ndvi_data = self.calculate_soil_moisture_index(ndvi_data)
        ndvi_data = self.analyze_ndvi_trend(ndvi_data)
        
        # Merge datasets on date and region
        combined_data = pd.merge(weather_data, ndvi_data, on=['date', 'region'], how='inner')
        
        # Compute overall climate risk
        combined_data = self.compute_climate_risk_score(combined_data)
        
        return combined_data

# Example usage
if __name__ == "__main__":
    # Initialize assessor
    assessor = ClimateRiskAssessor()
    
    # Generate sample data
    handler = DataHandler()
    weather_data = handler.generate_synthetic_weather_data()
    ndvi_data = handler.generate_synthetic_ndvi_data()
    
    # Assess climate risk
    risk_data = assessor.assess_climate_risk(weather_data, ndvi_data)
    
    # Display results
    print("Climate Risk Assessment Results:")
    print(risk_data[['date', 'region', 'climate_risk_score', 'climate_risk_level']].head(10))