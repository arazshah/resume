import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class RiskScoringModel:
    """Class to handle the risk scoring models and algorithms."""

    def __init__(self, model_type='random_forest'):
        """
        Initialize the risk scoring model.

        Args:
            model_type (str): Type of model to use. Options: 'random_forest', 'gradient_boosting', 'weighted_sum'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type != 'weighted_sum':
            logger.warning(
                f"Unknown model type: {model_type}. Defaulting to weighted_sum.")
            self.model_type = 'weighted_sum'

        logger.info(f"Initialized RiskScoringModel with {model_type}")

    def train(self, X, y):
        """
        Train the risk scoring model.

        Args:
            X (DataFrame or array-like): Features for training
            y (Series or array-like): Target values (historical risk scores)

        Returns:
            dict: Model performance metrics
        """
        if self.model_type == 'weighted_sum':
            logger.warning(
                "Cannot train weighted_sum model. Use set_weights() instead.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Get feature importance
        if self.model_type == 'random_forest':
            self.feature_importance = dict(
                zip(X.columns, self.model.feature_importances_))
        elif self.model_type == 'gradient_boosting':
            self.feature_importance = dict(
                zip(X.columns, self.model.feature_importances_))

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()

        metrics = {
            'mse': mse,
            'r2': r2,
            'cv_mse': cv_mse,
            'feature_importance': self.feature_importance
        }

        logger.info(f"Model trained. Test MSE: {mse:.4f}, R²: {r2:.4f}")
        return metrics

    def set_weights(self, weights):
        """
        Set weights for the weighted sum model.

        Args:
            weights (dict): Dictionary mapping feature names to weights

        Returns:
            bool: True if successful
        """
        if self.model_type != 'weighted_sum':
            logger.warning(f"Cannot set weights for {self.model_type} model.")
            return False

        self.weights = weights
        self.feature_importance = weights  # For consistency
        logger.info(f"Weights set for weighted_sum model: {weights}")
        return True

    def predict(self, X):
        """
        Predict risk scores for features X.

        Args:
            X (DataFrame or array-like): Features for prediction

        Returns:
            array-like: Predicted risk scores
        """
        if self.model_type == 'weighted_sum':
            if not hasattr(self, 'weights'):
                logger.error("Weights not set for weighted_sum model.")
                return None

            # Convert X to DataFrame if it's not already
            if not isinstance(X, pd.DataFrame):
                logger.warning(
                    "Converting X to DataFrame with default column names.")
                X = pd.DataFrame(X)

            # Check that all required features are present
            missing_features = set(self.weights.keys()) - set(X.columns)
            if missing_features:
                logger.error(
                    f"Missing features in input data: {missing_features}")
                return None

            # Calculate weighted sum
            weighted_sum = np.zeros(len(X))
            total_weight = sum(self.weights.values())

            for feature, weight in self.weights.items():
                weighted_sum += X[feature].values * weight

            risk_scores = weighted_sum / total_weight
            return risk_scores
        else:
            if self.model is None:
                logger.error("Model not trained.")
                return None

            return self.model.predict(X)

    def explain_score(self, X, index=0):
        """
        Explain the factors contributing to a specific risk score.

        Args:
            X (DataFrame): Features
            index (int): Index of the record to explain

        Returns:
            dict: Contribution of each feature to the risk score
        """
        if isinstance(X, pd.DataFrame):
            single_record = X.iloc[index].to_dict()
        else:
            logger.warning("X is not a DataFrame. Cannot explain score.")
            return None

        if self.model_type == 'weighted_sum':
            if not hasattr(self, 'weights'):
                logger.error("Weights not set for weighted_sum model.")
                return None

            contributions = {}
            total_weight = sum(self.weights.values())

            for feature, weight in self.weights.items():
                if feature in single_record:
                    contribution = (
                        single_record[feature] * weight) / total_weight
                    contributions[feature] = contribution

            return contributions
        elif self.feature_importance is not None:
            # This is a simplified approach for tree-based models
            contributions = {}
            for feature, importance in self.feature_importance.items():
                if feature in single_record:
                    # Scale by feature value and importance
                    contribution = single_record[feature] * importance
                    contributions[feature] = contribution

            # Normalize
            total = sum(contributions.values())
            if total != 0:
                for feature in contributions:
                    contributions[feature] /= total

            return contributions
        else:
            logger.warning(
                "Feature importance not available. Cannot explain score.")
            return None
