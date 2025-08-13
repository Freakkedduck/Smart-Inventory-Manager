"""
Sample Data Generator
=====================

This module generates realistic sample sales data for testing and demonstration
purposes. The generated data includes seasonality, promotions, and realistic
sales patterns across multiple stores and product families.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SalesDataGenerator:
    """
    Generates realistic sales data for multiple stores and product families.
    
    Features:
    - Seasonal patterns (daily, weekly, monthly, yearly)
    - Promotional effects
    - Store-specific variations
    - Product family differences
    - Realistic noise and outliers
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_data(self, 
                     start_date: str = '2021-01-01',
                     end_date: str = '2023-12-31',
                     stores: list = None,
                     product_families: list = None,
                     base_sales_range: tuple = (30, 80)) -> pd.DataFrame:
        """
        Generate comprehensive sales dataset.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        stores : list, optional
            List of store identifiers
        product_families : list, optional
            List of product family names
        base_sales_range : tuple
            Range for base daily sales (min, max)
            
        Returns:
        --------
        pd.DataFrame
            Generated sales data
        """
        print(f"Generating sales data from {start_date} to {end_date}...")
        
        # Default stores and families if not provided
        if stores is None:
            stores = [f'STORE_{i:02d}' for i in range(1, 6)]  # 5 stores
        
        if product_families is None:
            product_families = [
                'GROCERY', 'BEVERAGES', 'CLEANING', 'DAIRY', 
                'PRODUCE', 'MEAT', 'BAKERY'
            ]
        
        # Create date range
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Generate data
        data_rows = []
        
        for date in date_range:
            for store in stores:
                for family in product_families:
                    sales = self._generate_daily_sales(
                        date, store, family, base_sales_range
                    )
                    
                    # Generate promotion flag
                    promo = self._generate_promotion(date, store, family)
                    
                    data_rows.append({
                        'date': date,
                        'store_nbr': store,
                        'family': family,
                        'sales': round(sales, 2),
                        'onpromotion': promo
                    })
        
        df = pd.DataFrame(data_rows)
        print(f"Generated {len(df):,} records for {len(stores)} stores and {len(product_families)} product families")
        
        return df
    
    def _generate_daily_sales(self, date, store, family, base_sales_range):
        """Generate daily sales for a specific store-family-date combination."""
        
        # Base sales for this store-family combination
        base_sales = self._get_base_sales(store, family, base_sales_range)
        
        # Apply various effects
        seasonal_effect = self._get_seasonal_effect(date, family)
        day_of_week_effect = self._get_day_of_week_effect(date, family)
        holiday_effect = self._get_holiday_effect(date)
        store_effect = self._get_store_effect(store)
        trend_effect = self._get_trend_effect(date)
        
        # Random noise
        noise = np.random.normal(0, base_sales * 0.1)
        
        # Calculate final sales
        sales = (base_sales * 
                seasonal_effect * 
                day_of_week_effect * 
                holiday_effect * 
                store_effect * 
                trend_effect) + noise
        
        # Ensure non-negative sales
        return max(0, sales)
    
    def _get_base_sales(self, store, family, base_sales_range):
        """Get base sales level for store-family combination."""
        # Create consistent base sales using hash of store-family
        hash_val = hash(f"{store}_{family}") % 1000
        normalized_hash = hash_val / 1000
        
        min_sales, max_sales = base_sales_range
        base_sales = min_sales + (max_sales - min_sales) * normalized_hash
        
        # Product family adjustments
        family_multipliers = {
            'GROCERY': 1.2,
            'BEVERAGES': 1.1,
            'DAIRY': 0.9,
            'PRODUCE': 1.0,
            'MEAT': 0.8,
            'CLEANING': 0.6,
            'BAKERY': 0.7
        }
        
        multiplier = family_multipliers.get(family, 1.0)
        return base_sales * multiplier
    
    def _get_seasonal_effect(self, date, family):
        """Calculate seasonal effect based on month."""
        month = date.month
        
        # Different seasonal patterns by product family
        if family in ['PRODUCE']:
            # Strong summer peak
            seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
        elif family in ['BEVERAGES']:
            # Summer peak for beverages
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
        elif family in ['MEAT', 'BAKERY']:
            # Holiday peaks (November-December)
            if month in [11, 12]:
                seasonal = 1.3
            else:
                seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
        else:
            # General seasonal pattern
            seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)
        
        return max(0.5, seasonal)
    
    def _get_day_of_week_effect(self, date, family):
        """Calculate day of week effect."""
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend effect varies by product family
        if family in ['BEVERAGES', 'MEAT', 'BAKERY']:
            # Higher weekend sales
            weekend_multipliers = [0.9, 0.9, 0.95, 1.0, 1.1, 1.4, 1.3]  # Mon-Sun
        elif family in ['CLEANING']:
            # Lower weekend sales
            weekend_multipliers = [1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]  # Mon-Sun
        else:
            # Moderate weekend effect
            weekend_multipliers = [0.9, 0.95, 1.0, 1.0, 1.05, 1.2, 1.1]  # Mon-Sun
        
        return weekend_multipliers[day_of_week]
    
    def _get_holiday_effect(self, date):
        """Calculate holiday effect."""
        month = date.month
        day = date.day
        
        # Major holidays and seasons
        if month == 12 and day > 20:  # Christmas season
            return 1.5
        elif month == 11 and day > 20:  # Thanksgiving season
            return 1.3
        elif month == 7 and day == 4:  # July 4th
            return 1.2
        elif month == 1 and day == 1:  # New Year's Day
            return 0.6
        elif month == 12 and day == 25:  # Christmas Day
            return 0.4
        else:
            return 1.0
    
    def _get_store_effect(self, store):
        """Calculate store-specific effect."""
        # Different stores have different performance levels
        store_multipliers = {
            'STORE_01': 1.1,  # High performing store
            'STORE_02': 1.0,  # Average store
            'STORE_03': 0.9,  # Lower performing store
            'STORE_04': 1.05, # Above average
            'STORE_05': 0.95  # Below average
        }
        
        return store_multipliers.get(store, 1.0)
    
    def _get_trend_effect(self, date):
        """Calculate long-term trend effect."""
        # Slight upward trend over time
        start_date = datetime(2021, 1, 1)
        days_since_start = (date - start_date).days
        
        # 2% annual growth
        annual_growth_rate = 0.02
        daily_growth_rate = annual_growth_rate / 365
        
        return 1.0 + (days_since_start * daily_growth_rate)
    
    def _generate_promotion(self, date, store, family):
        """Generate promotion flag."""
        # Different promotion probabilities by product family
        promo_probabilities = {
            'GROCERY': 0.05,
            'BEVERAGES': 0.08,
            'CLEANING': 0.12,
            'DAIRY': 0.04,
            'PRODUCE': 0.06,
            'MEAT': 0.10,
            'BAKERY': 0.15
        }
        
        base_prob = promo_probabilities.get(family, 0.08)
        
        # Higher promotion probability during certain periods
        month = date.month
        if month in [11, 12]:  # Holiday season
            base_prob *= 1.5
        elif month in [6, 7, 8]:  # Summer season
            base_prob *= 1.2
        
        return np.random.random() < base_prob
    
    def generate_forecast_data(self, historical_df, forecast_days=30):
        """
        Generate mock forecast data based on historical patterns.
        
        Parameters:
        -----------
        historical_df : pd.DataFrame
            Historical sales data
        forecast_days : int
            Number of days to forecast
            
        Returns:
        --------
        pd.DataFrame
            Forecast data with actual, predicted, lower_bound, upper_bound
        """
        print(f"Generating forecast data for {forecast_days} days...")
        
        # Get the last date from historical data
        last_date = historical_df['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_data = []
        
        # Generate forecasts for each store-family combination
        for (store, family), group in historical_df.groupby(['store_nbr', 'family']):
            # Calculate recent average and trend
            recent_data = group.tail(30)  # Last 30 days
            avg_sales = recent_data['sales'].mean()
            sales_std = recent_data['sales'].std()
            
            # Simple trend calculation
            if len(recent_data) > 1:
                x = np.arange(len(recent_data))
                y = recent_data['sales'].values
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0
                intercept = avg_sales
            
            for i, date in enumerate(forecast_dates):
                # Generate seasonal effects similar to historical data
                seasonal_effect = self._get_seasonal_effect(date, family)
                day_effect = self._get_day_of_week_effect(date, family)
                holiday_effect = self._get_holiday_effect(date)
                
                # Base forecast with trend
                base_forecast = intercept + slope * (len(recent_data) + i)
                
                # Apply effects
                forecast = base_forecast * seasonal_effect * day_effect * holiday_effect
                
                # Add some uncertainty
                uncertainty = sales_std * 0.5
                lower_bound = forecast - 1.96 * uncertainty
                upper_bound = forecast + 1.96 * uncertainty
                
                # Generate "actual" values (for testing)
                actual = forecast + np.random.normal(0, sales_std * 0.3)
                actual = max(0, actual)
                
                forecast_data.append({
                    'date': date,
                    'store_nbr': store,
                    'family': family,
                    'actual': round(actual, 2),
                    'forecast': round(max(0, forecast), 2),
                    'lower_bound': round(max(0, lower_bound), 2),
                    'upper_bound': round(max(0, upper_bound), 2)
                })
        
        return pd.DataFrame(forecast_data)
    
    def add_external_factors(self, df, add_weather=True, add_events=True):
        """
        Add external factors to the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Sales data
        add_weather : bool
            Whether to add weather data
        add_events : bool
            Whether to add event data
            
        Returns:
        --------
        pd.DataFrame
            Enhanced dataset with external factors
        """
        enhanced_df = df.copy()
        
        if add_weather:
            # Generate mock weather data
            enhanced_df['temperature'] = np.random.normal(70, 15, len(df))
            enhanced_df['precipitation'] = np.random.exponential(0.1, len(df))
            
        if add_events:
            # Add mock events
            enhanced_df['local_event'] = np.random.choice(
                [True, False], len(df), p=[0.05, 0.95]
            )
            
        return enhanced_df
    
    def save_data(self, df, filename='sales_data.csv'):
        """Save generated data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
    def get_data_summary(self, df):
        """Get summary statistics of the generated data."""
        summary = {
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'total_sales': df['sales'].sum(),
            'avg_daily_sales': df['sales'].mean(),
            'stores': df['store_nbr'].nunique(),
            'product_families': df['family'].nunique(),
            'promotion_rate': df['onpromotion'].mean() * 100,
            'sales_distribution': df['sales'].describe()
        }
        
        return summary


def generate_sample_dataset(size='medium'):
    """
    Convenience function to generate standard sample datasets.
    
    Parameters:
    -----------
    size : str
        Dataset size: 'small', 'medium', 'large'
        
    Returns:
    --------
    pd.DataFrame
        Generated sales data
    """
    generator = SalesDataGenerator()
    
    if size == 'small':
        # 3 months, 3 stores, 4 families
        return generator.generate_data(
            start_date='2023-10-01',
            end_date='2023-12-31',
            stores=['STORE_01', 'STORE_02', 'STORE_03'],
            product_families=['GROCERY', 'BEVERAGES', 'DAIRY', 'CLEANING']
        )
    elif size == 'medium':
        # 1 year, 5 stores, 7 families (default)
        return generator.generate_data(
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    elif size == 'large':
        # 3 years, 10 stores, 10 families
        return generator.generate_data(
            start_date='2021-01-01',
            end_date='2023-12-31',
            stores=[f'STORE_{i:02d}' for i in range(1, 11)],
            product_families=[
                'GROCERY', 'BEVERAGES', 'CLEANING', 'DAIRY', 'PRODUCE', 
                'MEAT', 'BAKERY', 'FROZEN', 'HEALTH', 'ELECTRONICS'
            ]
        )
    else:
        raise ValueError("Size must be 'small', 'medium', or 'large'")


if __name__ == "__main__":
    # Generate and save sample data
    print("Generating sample sales data...")
    
    generator = SalesDataGenerator()
    df = generator.generate_data()
    
    # Print summary
    summary = generator.get_data_summary(df)
    print("\nDataset Summary:")
    for key, value in summary.items():
        if key != 'sales_distribution':
            print(f"{key}: {value}")
    
    # Save data
    generator.save_data(df, 'sample_sales_data.csv')
    print("\nSample data generation completed!")