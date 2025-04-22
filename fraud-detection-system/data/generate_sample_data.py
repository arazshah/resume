import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime, timedelta
import random
import string
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class FraudSampleGenerator:
    """Generate synthetic data for fraud detection testing."""

    def __init__(self, output_dir="fraud-detection-system/data"):
        """
        Initialize the sample data generator.

        Args:
            output_dir (str): Directory where sample data will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Default parameters
        self.num_claims = 1000
        self.fraud_percentage = 0.1  # 10% of claims will be fraudulent
        self.fraud_clusters = 3      # Number of geographic clusters of fraud
        self.time_period_days = 180  # 6 months of data

        # Locations for reference (US cities)
        self.cities = {
            "New York": {"lat": 40.7128, "lon": -74.0060},
            "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
            "Chicago": {"lat": 41.8781, "lon": -87.6298},
            "Houston": {"lat": 29.7604, "lon": -95.3698},
            "Phoenix": {"lat": 33.4484, "lon": -112.0740},
            "Philadelphia": {"lat": 39.9526, "lon": -75.1652},
            "San Antonio": {"lat": 29.4241, "lon": -98.4936},
            "San Diego": {"lat": 32.7157, "lon": -117.1611},
            "Dallas": {"lat": 32.7767, "lon": -96.7970},
            "San Jose": {"lat": 37.3382, "lon": -121.8863}
        }

        # Generate fraud clusters
        self.fraud_cluster_centers = self._generate_fraud_clusters()

    def _generate_fraud_clusters(self):
        """Generate random fraud cluster centers based on specified cities."""
        cities_list = list(self.cities.keys())
        selected_cities = random.sample(cities_list, min(
            self.fraud_clusters, len(cities_list)))

        clusters = []
        for city in selected_cities:
            center = self.cities[city]
            # Add some random offset from the city center
            cluster = {
                "city": city,
                "lat": center["lat"] + np.random.normal(0, 0.05),
                "lon": center["lon"] + np.random.normal(0, 0.05),
                "radius": np.random.uniform(0.02, 0.1)  # in degrees
            }
            clusters.append(cluster)

        return clusters

    def _generate_random_id(self, prefix, length=8):
        """Generate a random alphanumeric ID."""
        chars = string.ascii_uppercase + string.digits
        random_id = ''.join(random.choices(chars, k=length))
        return f"{prefix}-{random_id}"

    def _is_in_fraud_cluster(self, lat, lon):
        """Check if a location is within any fraud cluster."""
        for cluster in self.fraud_cluster_centers:
            dist = np.sqrt((lat - cluster["lat"])
                           ** 2 + (lon - cluster["lon"])**2)
            if dist < cluster["radius"]:
                return True, cluster["city"]
        return False, None

    def generate_claims_data(self, num_claims=None, fraud_percentage=None, time_period_days=None):
        """
        Generate synthetic claims data with fraud patterns.

        Args:
            num_claims (int, optional): Number of claims to generate
            fraud_percentage (float, optional): Percentage of fraudulent claims (0-1)
            time_period_days (int, optional): Time period for claims in days

        Returns:
            pandas.DataFrame: DataFrame with synthetic claims data
        """
        # Use provided parameters or defaults
        num_claims = num_claims or self.num_claims
        fraud_percentage = fraud_percentage or self.fraud_percentage
        time_period_days = time_period_days or self.time_period_days

        # Base date (today minus the time period)
        base_date = datetime.now() - timedelta(days=time_period_days)

        # Generate claim data
        data = []

        for i in range(num_claims):
            # Determine if this claim is fraudulent (based on percentage)
            is_fraud = random.random() < fraud_percentage

            # Generate claim date
            days_offset = int(np.random.uniform(0, time_period_days))
            claim_date = base_date + timedelta(days=days_offset)

            # Generate customer and policy IDs
            customer_id = self._generate_random_id("CUST")
            policy_id = self._generate_random_id("POL")

            # Generate location
            # If fraudulent, increase chance of being in a fraud cluster
            if is_fraud and random.random() < 0.7:
                # Pick a random fraud cluster
                cluster = random.choice(self.fraud_cluster_centers)
                # Generate location in that cluster with some noise
                lat = cluster["lat"] + np.random.normal(0, 0.02)
                lon = cluster["lon"] + np.random.normal(0, 0.02)
                city = cluster["city"]
            else:
                # Pick a random city
                city = random.choice(list(self.cities.keys()))
                center = self.cities[city]
                # Add noise
                lat = center["lat"] + np.random.normal(0, 0.1)
                lon = center["lon"] + np.random.normal(0, 0.1)

            # Check if location is in any fraud cluster (could be by chance)
            in_fraud_cluster, cluster_city = self._is_in_fraud_cluster(
                lat, lon)

            # Generate claim amount based on fraud status
            if is_fraud:
                # Fraudulent claims tend to be higher
                claim_amount = np.random.gamma(shape=10, scale=1000)
                # Sometimes fraudsters submit claims that are suspiciously round
                if random.random() < 0.3:
                    # Round to nearest thousand
                    claim_amount = round(claim_amount, -3)
            else:
                # Non-fraudulent claims follow a different distribution
                claim_amount = np.random.gamma(shape=5, scale=800)

            # Generate other claim attributes
            claim_type = random.choice(
                ['Auto', 'Home', 'Health', 'Property', 'Liability'])

            # Generate policyholder information
            first_name = random.choice(
                ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Jennifer', 'William', 'Elizabeth'])
            last_name = random.choice(
                ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor'])
            full_name = f"{first_name} {last_name}"

            # Generate address
            street_num = random.randint(100, 9999)
            street_name = random.choice(
                ['Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Washington', 'Park', 'Lake', 'Hill'])
            street_type = random.choice(
                ['St', 'Ave', 'Blvd', 'Dr', 'Ln', 'Rd', 'Way', 'Pl'])
            address = f"{street_num} {street_name} {street_type}"

            # Fraud indicators
            # Multiple claims from same customer in short time (for fraudsters)
            previous_claim = False
            days_since_previous = None

            if is_fraud and random.random() < 0.4:
                previous_claim = True
                days_since_previous = np.random.randint(7, 60)
            elif not is_fraud and random.random() < 0.1:
                previous_claim = True
                days_since_previous = np.random.randint(30, 180)

            # Claim status
            status = random.choice(
                ['Filed', 'Processing', 'Approved', 'Under Investigation', 'Rejected'])

            # Create claim record
            claim = {
                'claim_id': self._generate_random_id("CLM"),
                'customer_id': customer_id,
                'policy_id': policy_id,
                'policyholder_name': full_name,
                'claim_date': claim_date,
                'claim_amount': claim_amount,
                'claim_type': claim_type,
                'address': address,
                'city': city,
                'latitude': lat,
                'longitude': lon,
                'status': status,
                'previous_claim': previous_claim,
                'days_since_previous': days_since_previous,
                'in_fraud_cluster': in_fraud_cluster,
                'cluster_name': cluster_city,
                'is_fraud': is_fraud  # Ground truth for evaluation
            }

            data.append(claim)

        # Convert to DataFrame
        claims_df = pd.DataFrame(data)

        # Add some fraud patterns

        # 1. Same customer with multiple claims
        if num_claims > 100:
            # Take some fraudulent claims and assign them to the same customer
            fraud_indices = claims_df[claims_df['is_fraud']].index.tolist()

            if len(fraud_indices) >= 10:
                fraud_groups = min(5, len(fraud_indices) // 2)

                for i in range(fraud_groups):
                    # Pick 2-3 fraud claims and assign the same customer
                    group_size = random.randint(2, 3)
                    if len(fraud_indices) >= group_size:
                        group = random.sample(fraud_indices, group_size)
                        fraud_indices = [
                            idx for idx in fraud_indices if idx not in group]

                        # Assign same customer ID
                        shared_customer = self._generate_random_id("CUST")
                        claims_df.loc[group, 'customer_id'] = shared_customer

                        # Make the dates close together
                        base_date = claims_df.loc[group[0], 'claim_date']
                        for j, idx in enumerate(group[1:]):
                            claims_df.loc[idx, 'claim_date'] = base_date + \
                                timedelta(days=random.randint(5, 30))
                            claims_df.loc[idx, 'previous_claim'] = True
                            claims_df.loc[idx, 'days_since_previous'] = (
                                claims_df.loc[idx, 'claim_date'] - base_date).days

        # 2. Geographic patterns (already handled in generation)

        return claims_df

    def save_to_csv(self, claims_df, filename="sample_claims_data.csv"):
        """Save the generated claims data to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        claims_df.to_csv(filepath, index=False)
        print(f"Saved {len(claims_df)} sample claims to {filepath}")
        return filepath

    def save_to_shapefile(self, claims_df, filename="sample_claims_geodata.shp"):
        """Convert claims data to GeoDataFrame and save as shapefile."""
        # Create a GeoDataFrame from the claims data
        geometry = [Point(lon, lat) for lon, lat in zip(
            claims_df['longitude'], claims_df['latitude'])]
        gdf = gpd.GeoDataFrame(claims_df, geometry=geometry, crs="EPSG:4326")

        # Save to shapefile
        filepath = os.path.join(self.output_dir, filename)
        gdf.to_file(filepath)
        print(f"Saved {len(gdf)} sample claims as shapefile to {filepath}")
        return filepath

    def visualize_claims(self, claims_df, show_fraud=True, save_path=None):
        """
        Visualize the geographic distribution of claims.

        Args:
            claims_df (DataFrame): DataFrame with claim data
            show_fraud (bool): Whether to highlight fraudulent claims
            save_path (str, optional): Path to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Convert to GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(
            claims_df['longitude'], claims_df['latitude'])]
        gdf = gpd.GeoDataFrame(claims_df, geometry=geometry, crs="EPSG:4326")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all claims
        gdf.plot(ax=ax, color='blue', alpha=0.4, markersize=5)

        # Highlight fraudulent claims if requested
        if show_fraud:
            gdf[gdf['is_fraud']].plot(
                ax=ax, color='red', alpha=0.7, markersize=8)

            # Plot fraud clusters
            for cluster in self.fraud_cluster_centers:
                circle = plt.Circle((cluster['lon'], cluster['lat']), cluster['radius'],
                                    color='red', fill=False, alpha=0.5, linestyle='--')
                ax.add_patch(circle)
                ax.text(cluster['lon'], cluster['lat'], cluster['city'],
                        fontsize=10, ha='center', va='center', color='darkred')

        # Add base map
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs=gdf.crs.to_string())
        except ImportError:
            print("Contextily not installed. No basemap added.")

        # Adjust plot
        ax.set_title('Geographic Distribution of Insurance Claims')

        # Add legend
        if show_fraud:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, alpha=0.4, label='Normal Claims'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=8, alpha=0.7, label='Fraudulent Claims'),
                Line2D([0], [0], color='red', linestyle='--',
                       alpha=0.5, label='Fraud Clusters')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {save_path}")

        return fig

    def generate_and_save_all(self, visualize=True):
        """Generate, save, and optionally visualize all sample data."""
        # Generate claims data
        claims_df = self.generate_claims_data()

        # Save to CSV
        csv_path = self.save_to_csv(claims_df)

        # Save to shapefile
        shp_path = self.save_to_shapefile(claims_df)

        # Visualize
        if visualize:
            save_path = os.path.join(
                self.output_dir, "claims_visualization.png")
            self.visualize_claims(
                claims_df, show_fraud=True, save_path=save_path)

        return {
            'claims_df': claims_df,
            'csv_path': csv_path,
            'shp_path': shp_path,
            'visualize_path': save_path if visualize else None
        }


# Run the sample data generation if executed directly
if __name__ == "__main__":
    print("Generating sample fraud detection data...")
    generator = FraudSampleGenerator()
    results = generator.generate_and_save_all()

    # Print summary
    claims_df = results['claims_df']
    print("\nData Summary:")
    print(f"Total claims: {len(claims_df)}")
    print(
        f"Fraudulent claims: {claims_df['is_fraud'].sum()} ({claims_df['is_fraud'].mean() * 100:.1f}%)")
    print(
        f"Claims in fraud clusters: {claims_df['in_fraud_cluster'].sum()} ({claims_df['in_fraud_cluster'].mean() * 100:.1f}%)")

    print("\nClaim amounts:")
    print(f"Average amount: ${claims_df['claim_amount'].mean():.2f}")
    print(
        f"Average fraudulent claim: ${claims_df[claims_df['is_fraud']]['claim_amount'].mean():.2f}")
    print(
        f"Average legitimate claim: ${claims_df[~claims_df['is_fraud']]['claim_amount'].mean():.2f}")

    print("\nClaim types:")
    print(claims_df['claim_type'].value_counts())

    print("\nSample data generation complete!")
