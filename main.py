#!/usr/bin/env python3
"""
Main Execution Script
=====================

Run forecasting, inventory optimization, or launch dashboard via CLI.
"""

import argparse
import pandas as pd
from datetime import datetime
from forecasting import SalesForecastingModel, SimpleForecastingModel
from inventory_optimization import InventoryOptimizer, ABCAnalysis
from data_generator import SalesDataGenerator


def run_forecasting(data_file=None):
    print("=== SALES FORECASTING PIPELINE ===")
    # Load or generate data
    if data_file:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df):,} records from {data_file}")
    else:
        print("Generating sample data (medium size)...")
        df = SalesDataGenerator().generate_data()
    
    # Train model
    model = SimpleForecastingModel()
    model.fit(df)
    
    # Evaluate
    metrics = model.evaluate(df)
    print("Performance:", metrics)
    
    # Save predictions
    preds = model.predict(df)
    preds.to_csv('forecast_results.csv', index=False)
    print("Saved predictions → forecast_results.csv")


def run_inventory(data_file=None, service=0.95, lead=7):
    print("=== INVENTORY OPTIMIZATION PIPELINE ===")
    if data_file:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df):,} records from {data_file}")
    else:
        print("Generating sample data (medium size)...")
        df = SalesDataGenerator().generate_data()
    
    optimizer = InventoryOptimizer()
    results = optimizer.optimize_inventory(df, service_level=service, lead_time_days=lead)
    results.to_csv('inventory_optimization.csv', index=False)
    print("Results → inventory_optimization.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['forecast', 'inventory', 'dashboard'], default='dashboard')
    parser.add_argument('--data', help='CSV file of sales data')
    parser.add_argument('--service-level', type=float, default=0.95)
    parser.add_argument('--lead-time', type=int, default=7)
    args = parser.parse_args()

    if args.mode == 'forecast':
        run_forecasting(args.data)
    elif args.mode == 'inventory':
        run_inventory(args.data, args.service_level, args.lead_time)
    else:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])


if __name__ == '__main__':
    main()