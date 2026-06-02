import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_handler import DataHandler
from models.climate_risk import ClimateRiskAssessor
from models.pest_disease_risk import PestDiseaseRiskAssessor
from models.risk_assessment_system import AgriculturalRiskAssessmentSystem

def main():
    print("Testing Agricultural Risk Assessment System")
    print("=" * 50)
    
    # Initialize components
    data_handler = DataHandler()
    climate_assessor = ClimateRiskAssessor()
    pest_disease_assessor = PestDiseaseRiskAssessor()
    risk_system = AgriculturalRiskAssessmentSystem()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    weather_data = data_handler.generate_synthetic_weather_data('2023-01-01', '2023-12-31')
    ndvi_data = data_handler.generate_synthetic_ndvi_data('2023-01-01', '2023-12-31')
    pest_data = data_handler.generate_synthetic_pest_data('2023-01-01', '2023-12-31')
    price_data = data_handler.generate_synthetic_price_data('2023-01-01', '2023-12-31')
    
    print(f"Generated weather data for {weather_data['region'].nunique()} regions")
    print(f"Generated NDVI data for {ndvi_data['region'].nunique()} regions")
    print(f"Generated pest data for {pest_data['region'].nunique()} regions")
    print(f"Generated price data for {price_data['region'].nunique()} regions")
    
    # Save data
    print("\nSaving data to files...")
    data_handler.save_data(weather_data, 'weather_data.csv')
    data_handler.save_data(ndvi_data, 'ndvi_data.csv')
    data_handler.save_data(pest_data, 'pest_data.csv')
    data_handler.save_data(price_data, 'price_data.csv')
    
    # Test climate risk assessment
    print("\nAssessing climate risks...")
    climate_risk_data = climate_assessor.assess_climate_risk(weather_data, ndvi_data)
    print("Climate risk assessment completed")
    print(f"Climate risk scores range: {climate_risk_data['climate_risk_score'].min():.3f} - {climate_risk_data['climate_risk_score'].max():.3f}")
    
    # Test pest and disease risk assessment
    print("\nAssessing pest and disease risks...")
    pest_disease_risk_data = pest_disease_assessor.assess_pest_disease_risk(pest_data, weather_data)
    print("Pest and disease risk assessment completed")
    print(f"Pest/Disease risk scores range: {pest_disease_risk_data['pest_disease_risk_score'].min():.3f} - {pest_disease_risk_data['pest_disease_risk_score'].max():.3f}")
    
    # Run full assessment
    print("\nRunning full risk assessment...")
    assessment_results = risk_system.run_full_assessment(weather_data, ndvi_data, pest_data)
    print("Full risk assessment completed")
    
    # Display sample results
    print("\nSample Risk Assessment Results:")
    print(assessment_results[
        ['date', 'region', 'climate_risk_score', 'climate_risk_level', 
         'pest_disease_risk_score', 'pest_disease_risk_level',
         'total_risk_score', 'total_risk_level']
    ].head(10))
    
    # Generate report
    print("\nGenerating risk assessment report...")
    report = risk_system.generate_report(assessment_results)
    print("Risk Assessment Report Summary:")
    print(f"Assessment Date: {report['assessment_date']}")
    print(f"Regions Covered: {report['regions_covered']}")
    print(f"Average Climate Risk: {report['overall_stats']['avg_climate_risk']:.3f}")
    print(f"Average Pest/Disease Risk: {report['overall_stats']['avg_pest_disease_risk']:.3f}")
    print(f"Average Total Risk: {report['overall_stats']['avg_total_risk']:.3f}")
    print(f"High Risk Areas: {report['overall_stats']['high_risk_areas']}")
    
    # Save results
    print("\nSaving assessment results...")
    data_handler.save_data(assessment_results, 'risk_assessment_results.csv')
    print("Results saved successfully!")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()