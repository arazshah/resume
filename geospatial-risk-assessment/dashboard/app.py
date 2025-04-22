from src.risk_scoring import RiskScoringModel
from src.geospatial_processor import GeospatialProcessor
import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


app = Flask(__name__)

# Initialize processor and model
processor = GeospatialProcessor()
risk_model = RiskScoringModel(model_type='weighted_sum')

# Sample data paths - these would be configured properly in production
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
SAMPLE_SHAPEFILE = os.path.join(SAMPLE_DATA_DIR, 'sample_regions.shp')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default risk factor weights
DEFAULT_WEIGHTS = {
    'flood_risk': 2.0,
    'wildfire_risk': 1.5,
    'earthquake_risk': 1.8,
    'population_density': 1.0,
    'property_value': 1.2
}


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')


@app.route('/data')
def data():
    """Render the data management page."""
    return render_template('data.html')


@app.route('/analysis')
def analysis():
    """Render the risk analysis page."""
    return render_template('analysis.html')


@app.route('/api/load-sample-data', methods=['POST'])
def load_sample_data():
    """API endpoint to load sample data."""
    try:
        # Check if sample data exists
        if not os.path.exists(SAMPLE_SHAPEFILE):
            return jsonify({
                'success': False,
                'message': 'Sample data not found. Please upload data first.'
            }), 404

        # Load the shapefile
        gdf = processor.load_shapefile('regions', SAMPLE_SHAPEFILE)

        # Return basic stats about the data
        return jsonify({
            'success': True,
            'message': f'Loaded {len(gdf)} regions successfully',
            'feature_count': len(gdf),
            'columns': list(gdf.columns),
            'preview': gdf.drop(columns=['geometry']).head(5).to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading sample data: {str(e)}'
        }), 500


@app.route('/api/risk-factors', methods=['GET'])
def get_risk_factors():
    """API endpoint to get available risk factors."""
    # In a real system, this would be dynamic based on loaded data
    risk_factors = [
        {'id': 'flood_risk', 'name': 'Flood Risk',
            'weight': DEFAULT_WEIGHTS['flood_risk']},
        {'id': 'wildfire_risk', 'name': 'Wildfire Risk',
            'weight': DEFAULT_WEIGHTS['wildfire_risk']},
        {'id': 'earthquake_risk', 'name': 'Earthquake Risk',
            'weight': DEFAULT_WEIGHTS['earthquake_risk']},
        {'id': 'population_density', 'name': 'Population Density',
            'weight': DEFAULT_WEIGHTS['population_density']},
        {'id': 'property_value', 'name': 'Property Value',
            'weight': DEFAULT_WEIGHTS['property_value']}
    ]

    return jsonify({
        'success': True,
        'risk_factors': risk_factors
    })


@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    """API endpoint to calculate risk scores based on provided weights."""
    try:
        # Get weights from request
        data = request.json
        weights = data.get('weights', DEFAULT_WEIGHTS)

        # Load sample data if not already loaded
        if 'regions' not in processor.data_sources:
            gdf = processor.load_shapefile('regions', SAMPLE_SHAPEFILE)
        else:
            gdf = processor.data_sources['regions']

        # Set weights in risk model
        risk_model.set_weights(weights)

        # In a real system, we would have actual data for these factors
        # Here we'll simulate by generating random data
        np.random.seed(42)  # For reproducibility

        # Create a DataFrame with random risk factor data
        factor_data = {}
        for factor in weights.keys():
            # Generate random values between 0 and 1
            factor_data[factor] = np.random.random(len(gdf))

            # Add as risk factor to processor
            processor.add_risk_factor(factor, pd.Series(
                factor_data[factor]), weights[factor])

            # Normalize
            processor.normalize_factor(factor)

        # Create DataFrame for prediction
        X = pd.DataFrame(factor_data)

        # Calculate risk scores
        risk_scores = risk_model.predict(X)

        # Create a timestamped filename for the results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(
            RESULTS_DIR, f'risk_assessment_{timestamp}.csv')

        # Export results
        processor.export_risk_data(gdf, risk_scores, result_file)

        # Create a histogram of risk scores
        fig = px.histogram(
            risk_scores,
            nbins=20,
            labels={'value': 'Risk Score', 'count': 'Number of Regions'},
            title='Distribution of Risk Scores'
        )

        # Convert the plot to JSON
        histogram_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Return results
        return jsonify({
            'success': True,
            'message': f'Risk assessment completed. Results saved to {result_file}',
            'risk_scores': {
                'mean': float(np.mean(risk_scores)),
                'median': float(np.median(risk_scores)),
                'min': float(np.min(risk_scores)),
                'max': float(np.max(risk_scores)),
                'histogram': histogram_json
            },
            'result_file': result_file
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error calculating risk: {str(e)}'
        }), 500


@app.route('/api/explain-risk/<region_id>', methods=['GET'])
def explain_risk(region_id):
    """API endpoint to explain risk factors for a specific region."""
    try:
        region_id = int(region_id)

        # In a real system, we would look up the actual region
        # Here we'll simulate with random contribution data

        # Get risk factors and their weights
        factors = list(DEFAULT_WEIGHTS.keys())
        weights = list(DEFAULT_WEIGHTS.values())

        # Generate random contributions that sum to 1
        np.random.seed(region_id)  # Use region_id as seed for reproducibility
        raw_contributions = np.random.random(len(factors))
        contributions = raw_contributions / raw_contributions.sum()

        # Create explanation data
        explanation = []
        for i, factor in enumerate(factors):
            explanation.append({
                'factor': factor.replace('_', ' ').title(),
                'contribution': float(contributions[i]),
                'weight': float(weights[i])
            })

        # Sort by contribution (highest first)
        explanation.sort(key=lambda x: x['contribution'], reverse=True)

        # Create a bar chart of contributions
        fig = px.bar(
            explanation,
            x='factor',
            y='contribution',
            title=f'Risk Factor Contributions for Region {region_id}',
            labels={'factor': 'Risk Factor',
                    'contribution': 'Contribution to Overall Risk'}
        )

        # Convert the plot to JSON
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'region_id': region_id,
            'explanation': explanation,
            'chart': chart_json
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error explaining risk: {str(e)}'
        }), 500


@app.route('/results/<path:filename>')
def download_result(filename):
    """Endpoint to download result files."""
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
