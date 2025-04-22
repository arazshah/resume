import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeospatialProcessor:
    """Core class for processing geospatial data for risk assessment."""

    def __init__(self, config=None):
        """
        Initialize the GeospatialProcessor.

        Args:
            config (dict, optional): Configuration dictionary with processing parameters.
        """
        self.config = config or {}
        self.data_sources = {}
        self.risk_factors = {}
        self.scaler = MinMaxScaler()
        logger.info("GeospatialProcessor initialized")

    def load_shapefile(self, name, file_path):
        """
        Load a shapefile into GeoDataFrame.

        Args:
            name (str): Name identifier for the dataset
            file_path (str): Path to the shapefile

        Returns:
            GeoDataFrame: The loaded geospatial data
        """
        try:
            gdf = gpd.read_file(file_path)
            self.data_sources[name] = gdf
            logger.info(f"Loaded shapefile {name} with {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"Error loading shapefile {file_path}: {e}")
            raise

    def load_raster(self, name, file_path):
        """
        Load a raster dataset.

        Args:
            name (str): Name identifier for the dataset
            file_path (str): Path to the raster file

        Returns:
            DatasetReader: The loaded raster data
        """
        try:
            raster = rasterio.open(file_path)
            self.data_sources[name] = raster
            logger.info(f"Loaded raster {name}")
            return raster
        except Exception as e:
            logger.error(f"Error loading raster {file_path}: {e}")
            raise

    def add_risk_factor(self, name, data, weight=1.0):
        """
        Add a risk factor to the assessment model.

        Args:
            name (str): Name of the risk factor
            data (pandas.Series or numpy.ndarray): Risk factor data
            weight (float): Weight of this factor in the final score
        """
        self.risk_factors[name] = {
            'data': data,
            'weight': weight
        }
        logger.info(f"Added risk factor {name} with weight {weight}")

    def normalize_factor(self, name, invert=False):
        """
        Normalize a risk factor to a 0-1 scale.

        Args:
            name (str): Name of the risk factor to normalize
            invert (bool): If True, invert the values (1 - normalized value)
        """
        if name not in self.risk_factors:
            logger.error(f"Risk factor {name} not found")
            return

        data = self.risk_factors[name]['data'].values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data).flatten()

        if invert:
            normalized = 1 - normalized

        self.risk_factors[name]['normalized'] = normalized
        logger.info(f"Normalized risk factor {name}")

    def calculate_risk_scores(self):
        """
        Calculate the composite risk score based on all normalized risk factors.

        Returns:
            numpy.ndarray: Array of risk scores
        """
        total_weight = sum(factor['weight']
                           for factor in self.risk_factors.values())
        weighted_sum = np.zeros(
            len(next(iter(self.risk_factors.values()))['normalized']))

        for name, factor in self.risk_factors.items():
            if 'normalized' not in factor:
                logger.warning(
                    f"Risk factor {name} is not normalized and will be skipped")
                continue

            weighted_sum += factor['normalized'] * factor['weight']

        risk_scores = weighted_sum / total_weight
        logger.info("Risk scores calculated")
        return risk_scores

    def create_risk_map(self, gdf, risk_scores, output_path=None, column_name='risk_score'):
        """
        Create and save a risk heatmap.

        Args:
            gdf (GeoDataFrame): GeoDataFrame with geometries
            risk_scores (numpy.ndarray): Array of risk scores
            output_path (str, optional): Path to save the map image
            column_name (str): Column name for the risk scores

        Returns:
            folium.Map: Interactive risk map
        """
        # Add risk scores to the geodataframe
        gdf = gdf.copy()
        gdf[column_name] = risk_scores

        # Create a folium map
        center = [gdf.geometry.centroid.y.mean(
        ), gdf.geometry.centroid.x.mean()]
        risk_map = folium.Map(location=center, zoom_start=10)

        # Add choropleth layer
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=['id', column_name],
            key_on='feature.properties.id',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Risk Score',
        ).add_to(risk_map)

        if output_path:
            risk_map.save(output_path)
            logger.info(f"Risk map saved to {output_path}")

        return risk_map

    def export_risk_data(self, gdf, risk_scores, output_path, column_name='risk_score'):
        """
        Export the risk data to a CSV file.

        Args:
            gdf (GeoDataFrame): GeoDataFrame with geometries
            risk_scores (numpy.ndarray): Array of risk scores
            output_path (str): Path to save the CSV file
            column_name (str): Column name for the risk scores
        """
        export_df = gdf.copy()
        export_df[column_name] = risk_scores
        export_df.drop(columns=['geometry']).to_csv(output_path, index=False)
        logger.info(f"Risk data exported to {output_path}")
