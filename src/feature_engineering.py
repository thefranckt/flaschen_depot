"""
Feature Engineering Module
Processes raw data, creates features, and prepares data for model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from pathlib import Path

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
        df = service_times.copy()
        logger.info(f"Starting with service_times: {len(df)} rows")
        
        # Join with orders on web_order_id
        df = df.merge(orders, on='web_order_id', how='left', suffixes=('', '_order'))
        logger.info(f"After joining orders: {len(df)} rows")
        
        # Aggregate articles data by web_order_id
        articles_agg = self._aggregate_articles(articles)
        df = df.merge(articles_agg, on='web_order_id', how='left')
        logger.info(f"After joining articles: {len(df)} rows")
        
        # Driver mapping is already included via driver_id in service_times
        # But we can verify the mapping
        df = df.merge(
            driver_mapping[['driver_id', 'web_order_id']],
            on=['driver_id', 'web_order_id'],
            how='left',
            indicator=True
        )
        logger.info(f"Driver mapping verification complete")
        
        # Remove the merge indicator
        df = df.drop('_merge', axis=1)
        
        logger.info(f"Final merged dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    
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
        
        logger.info(f"Aggregated to {len(articles_agg)} orders")
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
        df = df.dropna(subset=['service_time_in_minutes'])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing target")
        
        # Remove negative or zero service times
        df = df[df['service_time_in_minutes'] > 0]
        logger.info(f"Removed rows with non-positive service times")
        
        # Handle missing values in key features
        # Floor: fill with 0 (ground floor)
        if 'floor' in df.columns:
            df['floor'] = df['floor'].fillna(0)
        
        # has_elevator: fill with False
        if 'has_elevator' in df.columns:
            df['has_elevator'] = df['has_elevator'].fillna(False)
        
        # is_business: fill with False
        if 'is_business' in df.columns:
            df['is_business'] = df['is_business'].fillna(False)
        
        # Fill missing article aggregations with 0
        article_cols = [
            'total_boxes', 'total_articles', 'total_weight_g',
            'avg_article_weight_g', 'max_article_weight_g', 'min_article_weight_g'
        ]
        for col in article_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Remove outliers using IQR method on target variable
        df = self._remove_outliers(df, 'service_time_in_minutes')
        
        logger.info(f"Data cleaning complete. Rows: {len(df)}")
        return df
    
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
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = initial_rows - len(df)
        
        logger.info(f"Removed {removed} outliers from {column} (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        return df
    
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
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
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
        
        logger.info(f"Prepared {len(feature_cols)} features and {len(y)} target values")
        logger.info(f"Feature columns: {feature_cols}")
        
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
    
    from data_loader import DataLoader
    
    # Test feature engineering
    loader = DataLoader()
    orders, articles, service_times, driver_mapping = loader.load_all()
    
    engineer = FeatureEngineer(random_state=42)
    X, y, feature_names, df = engineer.process_pipeline(
        orders, articles, service_times, driver_mapping
    )
    
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")
    print(f"Feature Names: {feature_names}")
    print(f"\nFeature Statistics:\n{X.describe()}")
    print(f"\nTarget Statistics:\n{y.describe()}")
