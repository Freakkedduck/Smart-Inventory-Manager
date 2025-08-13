"""
Sales Forecasting Module
========================

This module contains classes for sales forecasting using machine learning.
Includes feature engineering, model training, and prediction capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class SalesForecastingModel:
    """
    Advanced sales forecasting model with comprehensive feature engineering.
    Uses Gradient Boosting with time series features.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_names = []
        
    def create_time_features(self, df):
        """Create comprehensive time-based features"""
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Holiday indicators
        df['is_christmas_season'] = ((df['month'] == 12) & (df['day'] > 20)).astype(int)
        df['is_thanksgiving_season'] = ((df['month'] == 11) & (df['day'] > 20)).astype(int)
        
        return df
    
    def create_lag_features(self, df, target_col='sales', lags=[7, 14, 28]):
        """Create lag features for time series"""
        df = df.copy()
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])[target_col].shift(lag)
            
        return df
    
    def create_rolling_features(self, df, target_col='sales', windows=[7, 14, 28]):
        """Create rolling window statistics"""
        df = df.copy()
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            
        return df
    
    def prepare_features(self, df):
        """Prepare all features for modeling"""
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, lags=[7, 14, 28])
        
        # Create rolling features
        df = self.create_rolling_features(df, windows=[7, 14, 28])
        
        # Encode categorical variables
        categorical_cols = ['store_nbr', 'family']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature columns for modeling"""
        return [
            'year', 'month', 'day', 'dayofweek', 'quarter', 'weekofyear',
            'is_weekend', 'is_month_start', 'is_month_end',
            'is_christmas_season', 'is_thanksgiving_season', 'onpromotion',
            'store_nbr_encoded', 'family_encoded',
            'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_28'
        ]
    
    def train(self, df, target_col='sales'):
        """Train the forecasting model"""
        print("Preparing features for training...")
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        self.feature_names = feature_cols
        
        # Remove rows with NaN values (due to lag features)
        df_clean = df_features.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No data remaining after feature engineering. Check your data.")
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"Training on {len(X)} samples with {len(feature_cols)} features...")
        
        # Train Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.model.fit(X, y)
        
        print(f"Model training completed!")
        return self
    
    def predict(self, df):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        df_features = self.prepare_features(df)
        df_clean = df_features.dropna()
        
        if len(df_clean) == 0:
            return pd.DataFrame(columns=['date', 'store_nbr', 'family', 'sales', 'predictions'])
        
        X = df_clean[self.feature_names]
        predictions = self.model.predict(X)
        
        df_clean['predictions'] = predictions
        return df_clean[['date', 'store_nbr', 'family', 'sales', 'predictions']]
    
    def evaluate(self, df):
        """Evaluate model performance"""
        predictions = self.predict(df)
        
        if len(predictions) == 0:
            return {}
            
        mae = mean_absolute_error(predictions['sales'], predictions['predictions'])
        mape = mean_absolute_percentage_error(predictions['sales'], predictions['predictions']) * 100
        rmse = np.sqrt(mean_squared_error(predictions['sales'], predictions['predictions']))
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'samples': len(predictions)
        }


class SimpleForecastingModel:
    """
    Simplified forecasting model for faster computation.
    Uses Linear Regression with basic features per store-family combination.
    """
    
    def __init__(self):
        self.models = {}
        
    def fit(self, df):
        """Fit models for each store-family combination"""
        print("Training simple forecasting models...")
        
        model_count = 0
        for (store, family), group in df.groupby(['store_nbr', 'family']):
            if len(group) < 10:  # Skip if not enough data
                continue
                
            # Simple features
            group = group.sort_values('date')
            group['day_of_year'] = group['date'].dt.dayofyear
            group['month'] = group['date'].dt.month
            group['dayofweek'] = group['date'].dt.dayofweek
            group['promo_int'] = group['onpromotion'].astype(int)
            
            X = group[['day_of_year', 'month', 'dayofweek', 'promo_int']].values
            y = group['sales'].values
            
            model = LinearRegression()
            model.fit(X, y)
            self.models[(store, family)] = model
            model_count += 1
        
        print(f"Trained {model_count} models for store-family combinations")
        return self
        
    def predict(self, df):
        """Make predictions"""
        results = []
        for (store, family), group in df.groupby(['store_nbr', 'family']):
            if (store, family) in self.models:
                group = group.sort_values('date')
                group['day_of_year'] = group['date'].dt.dayofyear
                group['month'] = group['date'].dt.month
                group['dayofweek'] = group['date'].dt.dayofweek
                group['promo_int'] = group['onpromotion'].astype(int)
                
                X = group[['day_of_year', 'month', 'dayofweek', 'promo_int']].values
                predictions = self.models[(store, family)].predict(X)
                
                group = group.copy()
                group['predictions'] = predictions
                results.append(group[['date', 'store_nbr', 'family', 'sales', 'predictions']])
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame(columns=['date', 'store_nbr', 'family', 'sales', 'predictions'])
    
    def evaluate(self, df):
        """Evaluate model performance"""
        predictions = self.predict(df)
        
        if len(predictions) == 0:
            return {}
            
        mae = mean_absolute_error(predictions['sales'], predictions['predictions'])
        mape = mean_absolute_percentage_error(predictions['sales'], predictions['predictions']) * 100
        rmse = np.sqrt(mean_squared_error(predictions['sales'], predictions['predictions']))
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'samples': len(predictions)
        }


def cross_validate_model(model, df, n_splits=3):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Sort by date for proper time series splitting
    df_sorted = df.sort_values('date')
    
    scores = []
    for train_idx, test_idx in tscv.split(df_sorted):
        train_data = df_sorted.iloc[train_idx]
        test_data = df_sorted.iloc[test_idx]
        
        # Train model
        if hasattr(model, 'fit'):
            model.fit(train_data)
        else:
            model.train(train_data)
        
        # Evaluate
        metrics = model.evaluate(test_data)
        if metrics:
            scores.append(metrics)
    
    if scores:
        avg_scores = {
            'MAE': np.mean([s['MAE'] for s in scores]),
            'MAPE': np.mean([s['MAPE'] for s in scores]),
            'RMSE': np.mean([s['RMSE'] for s in scores])
        }
        return avg_scores
    else:
        return {}