#!/usr/bin/env python3
"""
GRA - Geospatial Risk Assessment Demo Script

This script demonstrates the core functionality of the Geospatial Risk Assessment system,
showing how it processes geographic data to evaluate insurance risks.
"""

from src.risk_scoring import RiskScoringModel
from src.geospatial_processor import GeospatialProcessor
import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import folium
from datetime import datetime

# Add parent directory to path to allow importing our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_regions(n_regions=50):
    """Create synthetic regions for demonstration purposes."""
    # Create random polygons as regions
    regions = []

    for i in range(n_regions):
        # Create a random center point
        center_x = np.random.uniform(0, 100)
        center_y = np.random.uniform(0, 100)

        # Create points around the center
        angles = np.linspace(0, 2*np.pi, 6)
        # Different radius for each angle
        radius = np.random.uniform(1, 5, size=6)

        # Calculate polygon vertices
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)

        # Create polygon
        polygon = Polygon(zip(x_coords, y_coords))

        # Add region data
        region = {
            'id': i,
            'name': f'Region {i}',
            'geometry': polygon
        }

        regions.append(region)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(regions)
    return gdf


def generate_risk_factors(gdf):
    """Generate synthetic risk factors for each region."""
    n_regions = len(gdf)

    # Create a dataframe for the risk factors
    risk_df = pd.DataFrame(index=gdf.index)

    # Generate flood risk based on y-coordinate (lower = higher risk)
    y_coords = np.array([geom.centroid.y for geom in gdf.geometry])
    risk_df['flood_risk'] = 1 - (y_coords / y_coords.max())

    # Generate wildfire risk based on x-coordinate (higher = higher risk)
    x_coords = np.array([geom.centroid.x for geom in gdf.geometry])
    risk_df['wildfire_risk'] = x_coords / x_coords.max()

    # Generate earthquake risk (random, but with spatial correlation)
    quake_center_x, quake_center_y = 25, 75
    distances = np.sqrt((x_coords - quake_center_x)**2 +
                        (y_coords - quake_center_y)**2)
    risk_df['earthquake_risk'] = 1 - (distances / distances.max())

    # Generate population density (random with some correlation to location)
    risk_df['population_density'] = 0.3 * risk_df['flood_risk'] + \
        0.2 * risk_df['wildfire_risk'] + 0.5 * np.random.random(n_regions)

    # Generate property values (somewhat inversely correlated with some risks)
    risk_df['property_value'] = 0.7 - 0.3 * risk_df['flood_risk'] + \
        0.2 * risk_df['earthquake_risk'] + 0.4 * np.random.random(n_regions)

    # Clip values to [0, 1] range
    risk_df = risk_df.clip(0, 1)

    # Add risk factors to the GeoDataFrame
    result_gdf = gdf.copy()
    for col in risk_df.columns:
        result_gdf[col] = risk_df[col]

    return result_gdf


def calculate_premiums(risk_scores, base_premium=1000, max_multiplier=3):
    """Calculate insurance premiums based on risk scores."""
    # Map risk score to a premium multiplier between 1 and max_multiplier
    multiplier = 1 + (max_multiplier - 1) * risk_scores
    premiums = base_premium * multiplier
    return premiums


def run_demo(output_dir='../results', n_regions=50, save_plots=True):
    """Run the GRA demonstration."""
    print("GRA - Geospatial Risk Assessment Demo")
    print("=====================================")

    # Create timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    print(f"\nGenerating {n_regions} sample regions...")
    regions_gdf = create_sample_regions(n_regions)

    print("Adding synthetic risk factors...")
    regions_with_risks = generate_risk_factors(regions_gdf)

    # Define risk factor weights
    weights = {
        'flood_risk': 2.0,       # Higher weight for flood risk
        'wildfire_risk': 1.5,
        'earthquake_risk': 1.8,
        'population_density': 1.0,
        'property_value': 1.2
    }

    # Initialize processor and model
    print("\nInitializing GRA system...")
    processor = GeospatialProcessor()
    risk_model = RiskScoringModel(model_type='weighted_sum')
    risk_model.set_weights(weights)

    # Add risk factors to processor
    risk_columns = list(weights.keys())
    for column in risk_columns:
        processor.add_risk_factor(
            column, regions_with_risks[column], weights[column])
        processor.normalize_factor(column)

    # Calculate risk scores
    print("Calculating risk scores...")
    X = regions_with_risks[risk_columns]
    risk_scores = risk_model.predict(X)
    regions_with_risks['risk_score'] = risk_scores

    # Calculate premiums
    print("Determining policy pricing...")
    regions_with_risks['premium'] = calculate_premiums(risk_scores)

    # Print summary statistics
    print("\nRisk Assessment Results:")
    print(f"  Mean Risk Score: {risk_scores.mean():.4f}")
    print(f"  Median Risk Score: {np.median(risk_scores):.4f}")
    print(f"  Min Risk Score: {risk_scores.min():.4f}")
    print(f"  Max Risk Score: {risk_scores.max():.4f}")
    print(f"  Standard Deviation: {risk_scores.std():.4f}")

    print("\nPremium Results:")
    print(f"  Mean Premium: ${regions_with_risks['premium'].mean():.2f}")
    print(f"  Median Premium: ${regions_with_risks['premium'].median():.2f}")
    print(f"  Min Premium: ${regions_with_risks['premium'].min():.2f}")
    print(f"  Max Premium: ${regions_with_risks['premium'].max():.2f}")

    # Find region with highest risk
    highest_risk_region_id = regions_with_risks['risk_score'].idxmax()
    highest_risk_region = regions_with_risks.loc[highest_risk_region_id]

    print(f"\nRegion with highest risk:")
    print(f"  ID: {highest_risk_region['id']}")
    print(f"  Risk Score: {highest_risk_region['risk_score']:.4f}")
    print(f"  Premium: ${highest_risk_region['premium']:.2f}")

    # Get risk factor contributions
    explanation = risk_model.explain_score(X, highest_risk_region_id)
    if explanation:
        print("\nRisk Factor Contributions:")
        for factor, contribution in sorted(explanation.items(), key=lambda x: x[1], reverse=True):
            print(f"  {factor.replace('_', ' ').title()}: {contribution:.4f} " +
                  f"({contribution / highest_risk_region['risk_score'] * 100:.1f}%)")

    # Save results to CSV
    output_file = os.path.join(output_dir, f'risk_assessment_{timestamp}.csv')
    processor.export_risk_data(
        regions_with_risks, risk_scores, output_file, 'risk_score')
    print(f"\nResults saved to: {output_file}")

    # Generate plots if requested
    if save_plots:
        # Risk score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(risk_scores, bins=15, alpha=0.7, color='blue')
        plt.axvline(risk_scores.mean(), color='red', linestyle='--',
                    label=f'Mean: {risk_scores.mean():.4f}')
        plt.title('Distribution of Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Number of Regions')
        plt.legend()
        plt.tight_layout()
        plot_file = os.path.join(
            output_dir, f'risk_distribution_{timestamp}.png')
        plt.savefig(plot_file)
        print(f"Risk distribution plot saved to: {plot_file}")

        # Create a risk map using Folium
        center = [50, 50]  # Center of our synthetic data
        risk_map = folium.Map(location=center, zoom_start=9)

        # Convert to GeoJSON for Folium
        geo_json = regions_with_risks.__geo_interface__

        # Add choropleth layer
        folium.Choropleth(
            geo_data=geo_json,
            data=regions_with_risks,
            columns=['id', 'risk_score'],
            key_on='feature.properties.id',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Risk Score',
        ).add_to(risk_map)

        # Save the map
        map_file = os.path.join(output_dir, f'risk_map_{timestamp}.html')
        risk_map.save(map_file)
        print(f"Interactive risk map saved to: {map_file}")

    print("\nDemo completed successfully!")
    return regions_with_risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GRA demonstration')
    parser.add_argument('--regions', type=int, default=50,
                        help='Number of sample regions to generate')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory for results and plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    run_demo(
        output_dir=args.output,
        n_regions=args.regions,
        save_plots=not args.no_plots
    )
