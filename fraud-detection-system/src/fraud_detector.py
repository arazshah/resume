import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud_detector')


class FraudDetector:
    """
    Machine learning-based fraud detection system for insurance claims.
    """

    def __init__(self, model_dir='fraud-detection-system/models',
                 results_dir='fraud-detection-system/results'):
        """
        Initialize the fraud detector.

        Args:
            model_dir (str): Directory to save/load trained models
            results_dir (str): Directory to save detection results
        """
        self.model_dir = model_dir
        self.results_dir = results_dir

        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Initialize model attributes
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.training_date = None

        # Default feature lists
        self.numeric_features = [
            'claim_amount',
            'latitude',
            'longitude',
            'days_since_previous'
        ]

        self.categorical_features = [
            'claim_type',
            'city',
            'status',
            'previous_claim'
        ]

        # Risk score thresholds
        self.risk_thresholds = {
            'high': 0.75,    # 75% or higher probability of fraud
            'medium': 0.5,   # 50-75% probability of fraud
            'low': 0.25      # 25-50% probability of fraud
        }

    def _prepare_preprocessor(self):
        """Create and return the feature preprocessing pipeline."""
        # Numeric feature preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical feature preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Column transformer that applies the appropriate preprocessing to each feature type
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])

        return preprocessor

    def _prepare_features(self, data):
        """
        Extract and prepare features from the claims data.

        Args:
            data (pandas.DataFrame): Claims data

        Returns:
            pandas.DataFrame: Processed features ready for the model
        """
        # Make a copy to avoid modifying the original
        df = data.copy()

        # Handle missing values in days_since_previous
        # (Replace NaN with 0 for claims with no previous claims)
        if 'days_since_previous' in df.columns:
            df['days_since_previous'] = df['days_since_previous'].fillna(0)

        # Convert boolean to int for preprocessing
        if 'previous_claim' in df.columns and df['previous_claim'].dtype == bool:
            df['previous_claim'] = df['previous_claim'].astype(int)

        # Convert date to datetime if it's not already
        if 'claim_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['claim_date']):
            df['claim_date'] = pd.to_datetime(df['claim_date'])

        # Extract additional features from date
        if 'claim_date' in df.columns:
            df['claim_month'] = df['claim_date'].dt.month
            df['claim_day_of_week'] = df['claim_date'].dt.dayofweek
            df['claim_day'] = df['claim_date'].dt.day

            # Add to numeric features if not already there
            for feat in ['claim_month', 'claim_day_of_week', 'claim_day']:
                if feat not in self.numeric_features:
                    self.numeric_features.append(feat)

        return df

    def train(self, training_data, target_col='is_fraud', test_size=0.2, random_state=42):
        """
        Train the fraud detection model.

        Args:
            training_data (pandas.DataFrame): Training data with fraudulent/non-fraudulent claims
            target_col (str): Name of the column containing the target (fraud indicator)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Training results including model performance metrics
        """
        logger.info("Starting model training...")

        # Prepare features
        df = self._prepare_features(training_data)

        # Split features and target
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in training data")

        X = df.copy()
        y = X.pop(target_col)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Create and fit preprocessor
        self.preprocessor = self._prepare_preprocessor()

        # Define the model - RandomForest is a good default
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        )

        # Create and fit the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])

        # Fit the model
        logger.info("Fitting model...")
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        self.training_date = datetime.now()

        # Evaluate on test set
        logger.info("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)

        # Get feature importance
        self._extract_feature_importance(pipeline)

        # Save the trained model
        self._save_model()

        # Save evaluation results
        eval_results = {
            'classification_report': clf_report,
            'confusion_matrix': conf_matrix.tolist(),
            'auc_score': auc_score,
            'train_test_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'fraud_train': sum(y_train),
                'fraud_test': sum(y_test)
            },
            'training_date': self.training_date.isoformat(),
            'feature_importance': self.feature_importance
        }

        # Save evaluation results
        results_file = os.path.join(
            self.results_dir, f"model_evaluation_{self.training_date.strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.info(
            f"Model trained successfully. Evaluation results saved to {results_file}")
        logger.info(f"Model AUC: {auc_score:.3f}")

        # Create and save evaluation visualizations
        self._create_evaluation_plots(y_test, y_pred, y_prob)

        return eval_results

    def _extract_feature_importance(self, pipeline):
        """Extract feature importance from the trained model."""
        # Get the feature names after preprocessing
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']

        # Get feature names from ColumnTransformer
        feature_names = []

        # Get the names for numeric features (they remain the same)
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    # For categorical features, we need to get the encoded feature names
                    encoder = trans.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        encoded_names = encoder.get_feature_names_out(cols)
                        feature_names.extend(encoded_names)

        # Get feature importance from the model
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_

            # Create feature importance dictionary
            if len(feature_names) == len(importance):
                # Create a dictionary of feature importance
                self.feature_importance = [{
                    'feature': str(feature),
                    'importance': float(imp)
                } for feature, imp in sorted(
                    zip(feature_names, importance),
                    key=lambda x: x[1],
                    reverse=True
                )]
            else:
                logger.warning(
                    "Feature names don't match feature importances length")
                self.feature_importance = [{'feature': f"feature_{i}", 'importance': float(imp)}
                                           for i, imp in enumerate(importance)]
        else:
            logger.warning("Model doesn't have feature_importances_ attribute")
            self.feature_importance = None

    def _create_evaluation_plots(self, y_test, y_pred, y_prob):
        """
        Create and save evaluation visualizations.

        Args:
            y_test (array): True labels
            y_pred (array): Predicted labels
            y_prob (array): Predicted probabilities
        """
        # Create a timestamp for unique filenames
        timestamp = self.training_date.strftime('%Y%m%d_%H%M%S')

        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_file = os.path.join(
            self.results_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_file, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Feature Importance
        if self.feature_importance:
            # Sort by importance and take top 15 features
            top_features = sorted(
                self.feature_importance, key=lambda x: x['importance'], reverse=True)[:15]

            plt.figure(figsize=(10, 8))
            feature_names = [f['feature'] for f in top_features]
            importances = [f['importance'] for f in top_features]

            # Create horizontal bar chart
            sns.barplot(x=importances, y=feature_names, palette='viridis')
            plt.title('Top 15 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()

            importance_file = os.path.join(
                self.results_dir, f"feature_importance_{timestamp}.png")
            plt.savefig(importance_file, bbox_inches='tight', dpi=300)
            plt.close()

        # 3. ROC Curve
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        roc_file = os.path.join(self.results_dir, f"roc_curve_{timestamp}.png")
        plt.savefig(roc_file, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Evaluation plots saved to {self.results_dir}")

    def _save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        # Create a timestamp for the model filename
        timestamp = self.training_date.strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(
            self.model_dir, f"fraud_detector_model_{timestamp}.pkl")

        # Save the model using pickle
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

        # Save model metadata
        metadata = {
            'model_file': model_file,
            'model_type': type(self.model.named_steps['classifier']).__name__,
            'training_date': self.training_date.isoformat(),
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_importance': self.feature_importance
        }

        metadata_file = os.path.join(
            self.model_dir, f"fraud_detector_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_file}")
        logger.info(f"Model metadata saved to {metadata_file}")

    def load_model(self, model_file=None):
        """
        Load a trained model from disk.

        Args:
            model_file (str, optional): Path to the model file. If None, the latest model will be loaded.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_file is None:
            # Find the latest model file
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith(
                'fraud_detector_model_') and f.endswith('.pkl')]

            if not model_files:
                logger.error("No model files found in model directory")
                return False

            # Sort by timestamp (which is part of the filename)
            model_files.sort(reverse=True)
            model_file = os.path.join(self.model_dir, model_files[0])

        # Load the model
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)

            # Load the corresponding metadata if it exists
            metadata_file = model_file.replace('model_', 'metadata_')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Extract relevant metadata
                self.training_date = datetime.fromisoformat(
                    metadata['training_date'])
                self.numeric_features = metadata.get(
                    'numeric_features', self.numeric_features)
                self.categorical_features = metadata.get(
                    'categorical_features', self.categorical_features)
                self.feature_importance = metadata.get('feature_importance')

                logger.info(f"Loaded model trained on {self.training_date}")
            else:
                logger.warning("Model metadata file not found")

            logger.info(f"Model loaded successfully from {model_file}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, claims_data):
        """
        Predict fraud probability for new claims.

        Args:
            claims_data (pandas.DataFrame): Claims data to predict on

        Returns:
            pandas.DataFrame: Original data with added fraud probability and risk level
        """
        if self.model is None:
            logger.error(
                "No trained model available. Train or load a model first.")
            return None

        logger.info(f"Predicting fraud for {len(claims_data)} claims")

        # Make a copy to avoid modifying the original
        result_df = claims_data.copy()

        # Prepare features
        df = self._prepare_features(claims_data)

        # Make predictions
        fraud_proba = self.model.predict_proba(df)[:, 1]

        # Add predictions to the result DataFrame
        result_df['fraud_probability'] = fraud_proba

        # Add risk level based on thresholds
        def risk_level(prob):
            if prob >= self.risk_thresholds['high']:
                return 'high'
            elif prob >= self.risk_thresholds['medium']:
                return 'medium'
            elif prob >= self.risk_thresholds['low']:
                return 'low'
            else:
                return 'very_low'

        result_df['risk_level'] = result_df['fraud_probability'].apply(
            risk_level)

        # Sort by fraud probability (highest first)
        result_df = result_df.sort_values('fraud_probability', ascending=False)

        return result_df

    def save_predictions(self, predictions_df, filename=None):
        """
        Save prediction results to disk.

        Args:
            predictions_df (pandas.DataFrame): DataFrame containing predictions
            filename (str, optional): Filename to save results. If None, a timestamp will be used.

        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fraud_predictions_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        predictions_df.to_csv(filepath, index=False)

        logger.info(f"Predictions saved to {filepath}")
        return filepath

    def get_high_risk_claims(self, predictions_df, threshold=None):
        """
        Extract high-risk claims from prediction results.

        Args:
            predictions_df (pandas.DataFrame): DataFrame containing predictions
            threshold (float, optional): Probability threshold. If None, the 'high' threshold is used.

        Returns:
            pandas.DataFrame: DataFrame containing only high-risk claims
        """
        threshold = threshold or self.risk_thresholds['high']

        high_risk = predictions_df[predictions_df['fraud_probability'] >= threshold].copy(
        )

        logger.info(
            f"Found {len(high_risk)} high-risk claims (threshold: {threshold})")
        return high_risk

    def evaluate_on_new_data(self, validation_data, true_label_col='is_fraud'):
        """
        Evaluate model performance on new data with known fraud labels.

        Args:
            validation_data (pandas.DataFrame): Validation data with fraud labels
            true_label_col (str): Name of the column containing true fraud labels

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error(
                "No trained model available. Train or load a model first.")
            return None

        if true_label_col not in validation_data.columns:
            logger.error(
                f"True label column '{true_label_col}' not found in validation data")
            return None

        # Make predictions
        predictions = self.predict(validation_data)

        # Calculate metrics
        y_true = validation_data[true_label_col]
        y_pred = predictions['fraud_probability'] >= self.risk_thresholds['medium']
        y_prob = predictions['fraud_probability']

        # Classification metrics
        clf_report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_prob)

        # Create evaluation visualizations
        self._create_evaluation_plots(y_true, y_pred, y_prob)

        # Compile results
        eval_results = {
            'classification_report': clf_report,
            'confusion_matrix': conf_matrix.tolist(),
            'auc_score': auc_score,
            'evaluation_date': datetime.now().isoformat(),
            'validation_size': len(validation_data),
            'fraud_count': int(sum(y_true)),
            'high_risk_count': int(sum(y_prob >= self.risk_thresholds['high'])),
            'medium_risk_count': int(sum((y_prob >= self.risk_thresholds['medium']) &
                                         (y_prob < self.risk_thresholds['high'])))
        }

        # Save evaluation results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            self.results_dir, f"validation_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Validation results saved to {results_file}")
        logger.info(f"Validation AUC: {auc_score:.3f}")

        return eval_results


# Example usage if run directly
if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

    from data.generate_sample_data import FraudSampleGenerator

    # Generate sample data
    print("Generating sample data for fraud detection...")
    generator = FraudSampleGenerator()
    claims_df = generator.generate_claims_data(num_claims=5000)

    # Split into train/test
    train_df, test_df = train_test_split(
        claims_df, test_size=0.3, random_state=42)

    # Create and train the fraud detector
    print("\nTraining fraud detection model...")
    detector = FraudDetector()
    eval_results = detector.train(train_df)

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_results = detector.evaluate_on_new_data(test_df)

    # Make predictions on new data
    print("\nMaking predictions on new claims...")
    predictions = detector.predict(test_df)

    # Get high-risk claims
    high_risk = detector.get_high_risk_claims(predictions)
    print(f"\nFound {len(high_risk)} high-risk claims")

    # Print a sample of high-risk claims
    if len(high_risk) > 0:
        print("\nSample high-risk claims:")
        sample = high_risk.head(5)
        for idx, row in sample.iterrows():
            print(f"Claim ID: {row.get('claim_id', 'N/A')}")
            print(f"Customer: {row.get('policyholder_name', 'N/A')}")
            print(f"Amount: ${row.get('claim_amount', 0):.2f}")
            print(f"Fraud Probability: {row['fraud_probability']:.3f}")
            print(f"Is Actually Fraud: {row.get('is_fraud', 'Unknown')}")
            print("-----")
