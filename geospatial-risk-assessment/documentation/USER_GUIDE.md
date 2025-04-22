# Geospatial Risk Assessment (GRA) System User Guide

## Introduction

The Geospatial Risk Assessment (GRA) system is a sophisticated tool designed for insurance underwriters to evaluate insurance risks and determine policy pricing based on geographic information. This guide will walk you through the system's key features and how to use them effectively.

## Getting Started

### System Requirements

- Python 3.8 or higher
- Required Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - geopandas
  - shapely
  - folium
  - plotly
  - flask
  - scikit-learn
  - matplotlib
  - seaborn

### Installation

1. Clone the repository or extract the project files to your local machine
2. Navigate to the project directory
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dashboard Usage

The GRA system provides an intuitive web-based dashboard for interactive risk analysis.

### Starting the Dashboard

1. Navigate to the project directory
2. Run the Flask application:
   ```
   cd dashboard
   python app.py
   ```
3. Open your web browser and go to: `http://localhost:5000`

### Dashboard Features

#### Data Loading
- Click the "Load Data" button to import geospatial data
- The system accepts shapefiles (.shp) with associated attribute data
- Sample data is included in the `data` directory

#### Risk Factor Configuration
- Adjust the weight sliders for each risk factor to customize the risk assessment model
- Weights range from 0 (no impact) to 5 (maximum impact)
- Common risk factors include:
  - Flood risk
  - Wildfire risk
  - Earthquake risk
  - Population density
  - Property value

#### Risk Calculation
- Click "Calculate Risk" to process the data with the selected weights
- The system will analyze the geospatial data and compute risk scores for each region
- Results will be displayed in the dashboard and can be exported for further analysis

#### Results Visualization
- Risk Score Distribution: Histogram showing the distribution of risk scores
- Risk Map: Interactive map displaying risk levels by region
- Risk Summary: Statistical overview of risk assessment results
- Detailed Breakdown: Individual risk factor contributions for each region

## Command-Line Interface

For batch processing or integration with other systems, the GRA provides a command-line interface.

### Running the Demo

```
cd src
python demo.py --regions 50 --output ../results
```

Options:
- `--regions`: Number of sample regions to generate (default: 50)
- `--output`: Directory to save results and plots (default: ../results)
- `--no-plots`: Skip generating visualization plots

### Output Files

The system generates several output files in the specified directory:
- `risk_assessment_[timestamp].csv`: Detailed risk data for all regions
- `risk_distribution_[timestamp].png`: Histogram of risk score distribution
- `risk_map_[timestamp].html`: Interactive map showing risk levels by region

## Programmatic API

For developers looking to integrate the GRA system into their applications, the system provides a Python API.

### Example Usage

```python
from geospatial_processor import GeospatialProcessor
from risk_scoring import RiskScoringModel

# Initialize processor and model
processor = GeospatialProcessor()
risk_model = RiskScoringModel(model_type='weighted_sum')

# Load geospatial data
regions = processor.load_shapefile('regions', 'path/to/regions.shp')

# Set risk factor weights
weights = {
    'flood_risk': 2.0,
    'wildfire_risk': 1.5,
    'earthquake_risk': 1.8,
    'population_density': 1.0,
    'property_value': 1.2
}
risk_model.set_weights(weights)

# Add risk factors to the processor
for factor, weight in weights.items():
    processor.add_risk_factor(factor, regions[factor], weight)
    processor.normalize_factor(factor)

# Calculate risk scores
risk_scores = risk_model.predict(regions[list(weights.keys())])

# Export results
processor.export_risk_data(regions, risk_scores, 'results.csv')

# Create visualization
risk_map = processor.create_risk_map(regions, risk_scores, 'risk_map.html')
```

## Advanced Features

### Custom Risk Scoring Models

The system supports different risk scoring models:
- `weighted_sum`: Simple weighted average of risk factors
- `random_forest`: Machine learning model that learns from historical data
- `gradient_boosting`: Advanced machine learning model for complex risk patterns

Example of training a machine learning model with historical data:

```python
from risk_scoring import RiskScoringModel

# Initialize model
model = RiskScoringModel(model_type='random_forest')

# Train with historical data
X = historical_data[risk_factors]  # Features
y = historical_data['historical_risk']  # Known risk values
metrics = model.train(X, y)

# View model performance
print(f"Model R² score: {metrics['r2']}")
print("Feature importance:", metrics['feature_importance'])

# Make predictions with the trained model
new_risk_scores = model.predict(new_data[risk_factors])
```

### Risk Explanation

The system can explain the factors contributing to a specific risk score:

```python
# Get explanation for a specific region
region_id = 42
explanation = risk_model.explain_score(X, region_id)

# Print contributions
for factor, contribution in sorted(explanation.items(), key=lambda x: x[1], reverse=True):
    print(f"{factor}: {contribution:.4f} ({contribution/risk_scores[region_id]*100:.1f}%)")
```

## Troubleshooting

### Common Issues

**Issue**: Dashboard doesn't load or shows errors  
**Solution**: Check that Flask is running and no other application is using port 5000

**Issue**: Error loading shapefile  
**Solution**: Ensure all related files (.dbf, .shx, etc.) are present in the same directory

**Issue**: Risk calculation fails  
**Solution**: Verify that your data includes all the necessary risk factor columns

### Getting Help

For additional assistance, please:
1. Check the API documentation in the `documentation` folder
2. Look for similar issues in the project repository
3. Contact the development team with detailed information about your issue

## Best Practices

1. **Data Quality**: Ensure your geospatial data is accurate and up-to-date
2. **Risk Factor Selection**: Choose risk factors that are relevant to your specific insurance domain
3. **Weight Calibration**: Periodically review and adjust risk factor weights based on actual claims data
4. **Validation**: Compare risk assessment results against historical claims to validate the model
5. **Documentation**: Keep records of risk assessment parameters for compliance and auditing purposes 