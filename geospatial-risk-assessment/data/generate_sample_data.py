#!/usr/bin/env python3
"""
Sample Data Generator for the GRA System

This script generates synthetic geospatial data for demonstrating the
Geospatial Risk Assessment system. It creates:
1. A shapefile with regions
2. Risk factor attributes for each region
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

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

    # Get centroids
    x_coords = np.array([geom.centroid.x for geom in gdf.geometry])
    y_coords = np.array([geom.centroid.y for geom in gdf.geometry])

    # Generate flood risk based on y-coordinate (lower = higher risk)
    risk_df['flood_risk'] = 1 - (y_coords / y_coords.max())

    # Generate wildfire risk based on x-coordinate (higher = higher risk)
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

    # Calculate a simple composite risk score for visualization
    risk_df['risk_score'] = (
        2.0 * risk_df['flood_risk'] +
        1.5 * risk_df['wildfire_risk'] +
        1.8 * risk_df['earthquake_risk'] +
        1.0 * risk_df['population_density'] +
        1.2 * risk_df['property_value']
    ) / 7.5  # Sum of weights

    # Clip values to [0, 1] range
    risk_df = risk_df.clip(0, 1)

    # Add risk factors to the GeoDataFrame
    result_gdf = gdf.copy()
    for col in risk_df.columns:
        result_gdf[col] = risk_df[col]

    return result_gdf


def main():
    """Generate sample data and save it to disk."""
    # Set random seed for reproducibility
    np.random.seed(42)

    print("Generating sample geospatial data for GRA system...")

    # Create sample regions
    print("Creating synthetic regions...")
    regions_gdf = create_sample_regions(50)

    # Add risk factors
    print("Generating risk factors...")
    regions_with_risks = generate_risk_factors(regions_gdf)

    # Save to shapefile
    output_file = os.path.join(os.path.dirname(__file__), 'sample_regions.shp')
    print(f"Saving data to {output_file}...")
    regions_with_risks.to_file(output_file)

    # Create a quick visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    regions_with_risks.plot(column='risk_score', ax=ax,
                            legend=True, cmap='YlOrRd')
    ax.set_title('Sample Regions with Risk Score')

    # Save the visualization
    viz_file = os.path.join(os.path.dirname(__file__),
                            'sample_visualization.png')
    plt.savefig(viz_file)

    print("Sample data generation complete!")
    print(
        f"Generated {len(regions_with_risks)} regions with the following attributes:")
    for col in regions_with_risks.columns:
        if col != 'geometry':
            print(f"- {col}")

    print(f"\nVisualization saved to {viz_file}")
    print("Use this data with the GRA system for demonstration and testing.")


if __name__ == "__main__":
    main()
