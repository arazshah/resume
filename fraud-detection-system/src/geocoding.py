import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
import os
import time
from geopy.geocoders import Nominatim, GoogleV3, ArcGIS
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import numpy as np
import h3
from tqdm import tqdm
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClaimGeocoder:
    """Class for geocoding insurance claim addresses."""

    def __init__(self, config=None):
        """
        Initialize the ClaimGeocoder.

        Args:
            config (dict, optional): Configuration parameters.
        """
        self.config = config or {
            'geocoder': 'nominatim',  # 'nominatim', 'google', or 'arcgis'
            'user_agent': 'fraud_detection_system',
            'google_api_key': os.environ.get('GOOGLE_API_KEY', None),
            'rate_limit': 1,  # seconds between consecutive requests
            'timeout': 10,     # seconds before timeout
            'max_retries': 3,  # number of retries for failed geocoding
            'batch_size': 100,  # number of records to process in one batch
            'h3_resolution': 8  # resolution for H3 hexagon indices
        }

        self._init_geocoder()
        logger.info("ClaimGeocoder initialized with geocoder: %s",
                    self.config['geocoder'])

    def _init_geocoder(self):
        """Initialize the geocoder based on configuration."""
        geocoder_type = self.config['geocoder'].lower()

        if geocoder_type == 'nominatim':
            self.geocoder = Nominatim(user_agent=self.config['user_agent'])
        elif geocoder_type == 'google' and self.config['google_api_key']:
            self.geocoder = GoogleV3(api_key=self.config['google_api_key'])
        elif geocoder_type == 'arcgis':
            self.geocoder = ArcGIS()
        else:
            # Default to Nominatim if invalid option
            logger.warning(
                "Invalid geocoder type or missing API key. Defaulting to Nominatim.")
            self.geocoder = Nominatim(user_agent=self.config['user_agent'])

        # Apply rate limiting
        self.geocode = RateLimiter(
            self.geocoder.geocode,
            min_delay_seconds=self.config['rate_limit'],
            error_wait_seconds=self.config['rate_limit'] * 2
        )

    def geocode_address(self, address, retries=None):
        """
        Geocode a single address.

        Args:
            address (str): The address to geocode.
            retries (int, optional): Number of retries. If None, use config value.

        Returns:
            tuple: (latitude, longitude) or (None, None) if geocoding failed.
        """
        if retries is None:
            retries = self.config['max_retries']

        attempt = 0
        while attempt <= retries:
            try:
                location = self.geocode(
                    address, timeout=self.config['timeout'])
                if location:
                    return (location.latitude, location.longitude)
                else:
                    logger.warning(
                        f"No geocoding result for address: {address}")
                    return (None, None)
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                attempt += 1
                if attempt <= retries:
                    wait_time = self.config['rate_limit'] * (2 ** attempt)
                    logger.warning(
                        f"Geocoding error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to geocode after {retries} retries: {address}")
                    return (None, None)
            except Exception as e:
                logger.error(f"Unexpected geocoding error: {e}")
                return (None, None)

    def geocode_dataframe(self, df, address_col, lat_col='latitude', lon_col='longitude', progress=True):
        """
        Geocode addresses in a DataFrame.

        Args:
            df (DataFrame): DataFrame containing addresses
            address_col (str): Column name with addresses
            lat_col (str): Column name to store latitude
            lon_col (str): Column name to store longitude
            progress (bool): Whether to show progress bar

        Returns:
            DataFrame: DataFrame with added latitude and longitude columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Initialize output columns
        if lat_col not in result_df.columns:
            result_df[lat_col] = np.nan

        if lon_col not in result_df.columns:
            result_df[lon_col] = np.nan

        # Get addresses to geocode (skip if lat/lon already exists)
        to_geocode = result_df[
            (result_df[address_col].notna()) &
            (result_df[lat_col].isna() | result_df[lon_col].isna())
        ]

        if len(to_geocode) == 0:
            logger.info("No addresses to geocode")
            return result_df

        logger.info(
            f"Geocoding {len(to_geocode)} addresses using {self.config['geocoder']}")

        # Process in batches to allow better error handling
        batch_size = min(self.config['batch_size'], len(to_geocode))
        batches = np.array_split(
            to_geocode.index, np.ceil(len(to_geocode) / batch_size))

        if progress:
            batches = tqdm(batches, desc="Geocoding")

        for batch_indices in batches:
            for idx in batch_indices:
                address = result_df.loc[idx, address_col]
                if pd.isna(address) or address == '':
                    continue

                lat, lon = self.geocode_address(address)

                if lat is not None and lon is not None:
                    result_df.loc[idx, lat_col] = lat
                    result_df.loc[idx, lon_col] = lon

        # Count success rate
        geocoded_count = result_df[(result_df[lat_col].notna()) & (
            result_df[lon_col].notna())].shape[0]
        success_rate = geocoded_count / \
            len(to_geocode) if len(to_geocode) > 0 else 0

        logger.info(
            f"Geocoded {geocoded_count} addresses ({success_rate:.1%} success rate)")
        return result_df

    def convert_to_geodataframe(self, df, lat_col='latitude', lon_col='longitude', crs="EPSG:4326"):
        """
        Convert a DataFrame with lat/lon columns to a GeoDataFrame.

        Args:
            df (DataFrame): DataFrame with latitude and longitude columns
            lat_col (str): Name of latitude column
            lon_col (str): Name of longitude column
            crs (str): Coordinate reference system

        Returns:
            GeoDataFrame: GeoDataFrame with Point geometry
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Filter to rows with valid coordinates
        valid_coords = result_df[(result_df[lat_col].notna()) & (
            result_df[lon_col].notna())]

        if len(valid_coords) == 0:
            logger.warning(
                "No valid coordinates found - returning original DataFrame")
            return df

        # Create geometry column
        geometry = [Point(xy) for xy in zip(
            valid_coords[lon_col], valid_coords[lat_col])]

        # Create GeoDataFrame
        try:
            gdf = gpd.GeoDataFrame(valid_coords, geometry=geometry, crs=crs)
            logger.info(f"Created GeoDataFrame with {len(gdf)} points")

            # Add H3 indices
            gdf['h3_index'] = gdf.apply(
                lambda row: h3.geo_to_h3(
                    row.geometry.y, row.geometry.x, self.config['h3_resolution']),
                axis=1
            )

            return gdf
        except Exception as e:
            logger.error(f"Error creating GeoDataFrame: {e}")
            return df

    def reverse_geocode(self, lat, lon, retries=None):
        """
        Reverse geocode coordinates to get address.

        Args:
            lat (float): Latitude
            lon (float): Longitude
            retries (int, optional): Number of retries. If None, use config value.

        Returns:
            str: Address or None if reverse geocoding failed
        """
        if retries is None:
            retries = self.config['max_retries']

        attempt = 0
        while attempt <= retries:
            try:
                location = self.geocoder.reverse(
                    (lat, lon), timeout=self.config['timeout'])
                if location:
                    return location.address
                else:
                    logger.warning(
                        f"No reverse geocoding result for ({lat}, {lon})")
                    return None
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                attempt += 1
                if attempt <= retries:
                    wait_time = self.config['rate_limit'] * (2 ** attempt)
                    logger.warning(
                        f"Reverse geocoding error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to reverse geocode after {retries} retries: ({lat}, {lon})")
                    return None
            except Exception as e:
                logger.error(f"Unexpected reverse geocoding error: {e}")
                return None

    def h3_cluster_analysis(self, gdf, value_col=None):
        """
        Analyze claim density in H3 hexagons.

        Args:
            gdf (GeoDataFrame): GeoDataFrame with h3_index column
            value_col (str, optional): Column to aggregate (count if None)

        Returns:
            DataFrame: H3 hexagon statistics
        """
        if 'h3_index' not in gdf.columns:
            logger.warning(
                "H3 index column not found. Converting to GeoDataFrame first.")
            gdf = self.convert_to_geodataframe(gdf)

        if 'h3_index' not in gdf.columns:
            logger.error("Failed to create H3 indices.")
            return None

        # Group by H3 index
        if value_col is not None and value_col in gdf.columns:
            h3_stats = gdf.groupby('h3_index')[value_col].agg(
                ['count', 'sum', 'mean'])
            h3_stats.columns = ['claim_count',
                                f'{value_col}_sum', f'{value_col}_mean']
        else:
            h3_stats = gdf.groupby('h3_index').size(
            ).reset_index(name='claim_count')

        # Add geographic coordinates for each hexagon
        h3_stats['latitude'] = h3_stats.index.map(lambda h: h3.h3_to_geo(h)[0])
        h3_stats['longitude'] = h3_stats.index.map(
            lambda h: h3.h3_to_geo(h)[1])

        # Calculate density (claims per square km)
        # Area of H3 hexagon at resolution 8 is approximately 0.74 km²
        h3_resolution = self.config['h3_resolution']
        hexagon_area = h3.hex_area(h3_resolution, 'km^2')
        h3_stats['density'] = h3_stats['claim_count'] / hexagon_area

        logger.info(
            f"Analyzed {len(gdf)} claims across {len(h3_stats)} H3 hexagons")
        return h3_stats

    def calculate_distances(self, gdf, point_or_address, address_is_string=True):
        """
        Calculate distances from claims to a reference point.

        Args:
            gdf (GeoDataFrame): GeoDataFrame with geometries
            point_or_address: Reference point (lat, lon) tuple or address string
            address_is_string (bool): Whether the point_or_address is a string address

        Returns:
            Series: Distances in kilometers
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            logger.warning("Input is not a GeoDataFrame. Converting first.")
            gdf = self.convert_to_geodataframe(gdf)

        if len(gdf) == 0:
            logger.warning("Empty GeoDataFrame - no distances to calculate")
            return pd.Series()

        # Get reference point coordinates
        ref_lat, ref_lon = None, None

        if address_is_string:
            ref_lat, ref_lon = self.geocode_address(point_or_address)
            if ref_lat is None:
                logger.error(
                    f"Could not geocode reference address: {point_or_address}")
                return pd.Series(index=gdf.index)
        else:
            ref_lat, ref_lon = point_or_address

        # Calculate distances using geodesic
        distances = gdf.apply(
            lambda row: geodesic((ref_lat, ref_lon),
                                 (row.geometry.y, row.geometry.x)).kilometers
            if not pd.isna(row.geometry) else np.nan,
            axis=1
        )

        logger.info(
            f"Calculated {len(distances)} distances from reference point ({ref_lat}, {ref_lon})")
        return distances
