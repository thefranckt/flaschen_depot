"""
Feature Engineering Module
Processes raw data, creates features, and prepares data for model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles data joining, cleaning, and feature engineering."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureEngineer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def join_datasets(
        self, 
        orders: pd.DataFrame,
        articles: pd.DataFrame,
        service_times: pd.DataFrame,
        driver_mapping: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join all datasets on appropriate keys.
        
        Args:
            orders: Orders DataFrame
            articles: Articles DataFrame
            service_times: Service times DataFrame
            driver_mapping: Driver mapping DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Starting dataset join...")
        
        # Start with service_times as it contains the target variable
        df_merged = service_times.copy()
        logger.info("Starting with service_times: %d rows", len(df_merged))
        
        # Join with orders on web_order_id
        df_merged = df_merged.merge(orders, on='web_order_id', how='left', suffixes=('', '_order'))
        logger.info("After joining orders: %d rows", len(df_merged))
        
        # Aggregate articles data by web_order_id
        articles_agg = self._aggregate_articles(articles)
        df_merged = df_merged.merge(articles_agg, on='web_order_id', how='left')
        logger.info("After joining articles: %d rows", len(df_merged))
        
        # Driver mapping is already included via driver_id in service_times
        # But we can verify the mapping
        df_merged = df_merged.merge(
            driver_mapping[['driver_id', 'web_order_id']],
            on=['driver_id', 'web_order_id'],
            how='left',
            indicator=True
        )
        logger.info("Driver mapping verification complete")
        
        # Remove the merge indicator
        df_merged = df_merged.drop('_merge', axis=1)
        
        logger.info("Final merged dataset: %d rows, %d columns", len(df_merged), len(df_merged.columns))
        return df_merged
    
    def _aggregate_articles(self, articles: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate article-level data to order level.
        
        Args:
            articles: Articles DataFrame
            
        Returns:
            Aggregated DataFrame with one row per web_order_id
        """
        logger.info("Aggregating articles data...")
        
        agg_dict = {
            'box_id': 'nunique',  # number of unique boxes
            'article_id': 'count',  # total number of articles
            'article_weight_in_g': ['sum', 'mean', 'max', 'min']
        }
        
        articles_agg = articles.groupby('web_order_id').agg(agg_dict).reset_index()
        
        # Flatten column names
        articles_agg.columns = [
            'web_order_id',
            'total_boxes',
            'total_articles',
            'total_weight_g',
            'avg_article_weight_g',
            'max_article_weight_g',
            'min_article_weight_g'
        ]
        
        logger.info("Aggregated to %d orders", len(articles_agg))
        return articles_agg
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and removing outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df)
        
        # Remove rows with missing target variable
        df_clean = df.dropna(subset=['service_time_in_minutes'])
        logger.info("Removed %d rows with missing target", initial_rows - len(df_clean))
        
        # Remove negative or zero service times
        df_clean = df_clean[df_clean['service_time_in_minutes'] > 0]
        logger.info("Removed rows with non-positive service times")
        
        # Handle missing values in key features
        # Floor: fill with 0 (ground floor)
        if 'floor' in df_clean.columns:
            df_clean['floor'] = df_clean['floor'].fillna(0)
        
        # has_elevator: fill with False
        if 'has_elevator' in df_clean.columns:
            df_clean['has_elevator'] = df_clean['has_elevator'].fillna(False)
        
        # is_business: fill with False
        if 'is_business' in df_clean.columns:
            df_clean['is_business'] = df_clean['is_business'].fillna(False)
        
        # Fill missing article aggregations with 0
        article_cols = [
            'total_boxes', 'total_articles', 'total_weight_g',
            'avg_article_weight_g', 'max_article_weight_g', 'min_article_weight_g'
        ]
        for col in article_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Remove outliers using IQR method on target variable
        df_clean = self._remove_outliers(df_clean, 'service_time_in_minutes')
        
        logger.info("Data cleaning complete. Rows: %d", len(df_clean))
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame, column: str, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            iqr_multiplier: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        initial_rows = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = initial_rows - len(df_filtered)
        
        logger.info("Removed %d outliers from %s (bounds: %.2f - %.2f)", removed, column, lower_bound, upper_bound)
        return df_filtered
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.
        
        Args:
            df: Input DataFrame with joined data
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Creating engineered features...")
        
        # Convert datetime columns
        if 'order_time' in df.columns:
            df['order_time'] = pd.to_datetime(df['order_time'])
            df['order_hour'] = df['order_time'].dt.hour
            df['order_day_of_week'] = df['order_time'].dt.dayofweek
            df['order_month'] = df['order_time'].dt.month
            df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
        
        # Weight per box
        df['weight_per_box'] = np.where(
            df['total_boxes'] > 0,
            df['total_weight_g'] / df['total_boxes'],
            0
        )
        
        # Floor and elevator interaction
        df['floor_elevator_interaction'] = df['floor'] * df['has_elevator'].astype(int)
        
        # Business and floor interaction
        df['business_floor_interaction'] = df['is_business'].astype(int) * df['floor']
        
        # Convert booleans to int for modeling
        bool_cols = ['has_elevator', 'is_business']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        logger.info("Feature engineering complete. Total features: %d", len(df.columns))
        return df
    
    def prepare_features_and_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare feature matrix and target variable.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series, feature names list)
        """
        logger.info("Preparing features and target...")
        
        # Define feature columns
        feature_cols = [
            # Order features
            'floor',
            'has_elevator',
            'is_business',
            'order_hour',
            'order_day_of_week',
            'order_month',
            'is_weekend',
            
            # Article aggregation features
            'total_boxes',
            'total_articles',
            'total_weight_g',
            'avg_article_weight_g',
            'max_article_weight_g',
            'min_article_weight_g',
            'weight_per_box',
            
            # Interaction features
            'floor_elevator_interaction',
            'business_floor_interaction',
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df['service_time_in_minutes'].copy()
        
        logger.info("Prepared %d features and %d target values", len(feature_cols), len(y))
        logger.info("Feature columns: %s", feature_cols)
        
        return X, y, feature_cols
    
    def process_pipeline(
        self,
        orders: pd.DataFrame,
        articles: pd.DataFrame,
        service_times: pd.DataFrame,
        driver_mapping: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
        """
        Execute full feature engineering pipeline.
        
        Args:
            orders: Orders DataFrame
            articles: Articles DataFrame
            service_times: Service times DataFrame
            driver_mapping: Driver mapping DataFrame
            
        Returns:
            Tuple of (features, target, feature_names, full_processed_df)
        """
        logger.info("Starting full feature engineering pipeline...")
        
        # Join datasets
        df = self.join_datasets(orders, articles, service_times, driver_mapping)
        
        # Clean data
        df = self.clean_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        
        logger.info("Feature engineering pipeline complete")
        return X, y, feature_names, df


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.data_loader import DataLoader
    
    # Test feature engineering
    loader = DataLoader()
    test_orders, test_articles, test_service_times, test_driver_mapping = loader.load_all()
    
    engineer = FeatureEngineer(random_state=42)
    X_result, y_result, feature_names_result, df_result = engineer.process_pipeline(
        test_orders, test_articles, test_service_times, test_driver_mapping
    )
    
    print(f"\nFeature Matrix Shape: {X_result.shape}")
    print(f"Target Shape: {y_result.shape}")
    print(f"Feature Names: {feature_names_result}")
    print(f"\nFeature Statistics:\n{X_result.describe()}")
    print(f"\nTarget Statistics:\n{y_result.describe()}")
