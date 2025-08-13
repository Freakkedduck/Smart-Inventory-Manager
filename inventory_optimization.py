"""
Inventory Optimization Module
=============================

This module contains classes for inventory optimization including safety stock 
calculation, reorder point determination, and inventory policy simulation.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """
    Comprehensive inventory optimization class.
    
    Calculates optimal safety stock, reorder points, and order quantities
    based on demand forecasts and service level requirements.
    """
    
    def __init__(self):
        self.results = None
        self.demand_stats = {}
        
    def calculate_demand_stats(self, historical_data: pd.DataFrame, 
                             store: str, family: str, 
                             lead_time_days: int = 7) -> Optional[Dict]:
        """
        Calculate demand statistics for a specific store-family combination.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical sales data with columns: date, store_nbr, family, sales
        store : str
            Store identifier
        family : str
            Product family identifier
        lead_time_days : int
            Lead time in days
            
        Returns:
        --------
        Dict or None
            Dictionary containing demand statistics or None if no data
        """
        # Filter data for specific store-family
        data = historical_data[
            (historical_data['store_nbr'] == store) & 
            (historical_data['family'] == family)
        ].copy()
        
        if len(data) == 0:
            return None
            
        # Calculate basic statistics
        daily_demand_mean = data['sales'].mean()
        daily_demand_std = data['sales'].std()
        
        # Handle edge case where std is 0 or very small
        if daily_demand_std < 0.01 or np.isnan(daily_demand_std):
            daily_demand_std = daily_demand_mean * 0.1  # Assume 10% CV
        
        # Calculate lead time demand statistics
        lead_time_demand_mean = daily_demand_mean * lead_time_days
        lead_time_demand_std = daily_demand_std * np.sqrt(lead_time_days)
        
        return {
            'daily_demand_mean': daily_demand_mean,
            'daily_demand_std': daily_demand_std,
            'lead_time_demand_mean': lead_time_demand_mean,
            'lead_time_demand_std': lead_time_demand_std,
            'lead_time_days': lead_time_days,
            'data_points': len(data),
            'min_demand': data['sales'].min(),
            'max_demand': data['sales'].max()
        }
    
    def calculate_safety_stock(self, demand_stats: Dict, 
                             service_level: float = 0.95) -> float:
        """
        Calculate safety stock based on service level using normal distribution.
        
        Parameters:
        -----------
        demand_stats : Dict
            Demand statistics from calculate_demand_stats
        service_level : float
            Target service level (0-1)
            
        Returns:
        --------
        float
            Safety stock quantity
        """
        if demand_stats is None:
            return 0
            
        # Z-score for the service level
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock = Z-score * standard deviation of demand during lead time
        safety_stock = z_score * demand_stats['lead_time_demand_std']
        
        return max(0, safety_stock)
    
    def calculate_reorder_point(self, demand_stats: Dict, 
                              safety_stock: float) -> float:
        """
        Calculate reorder point.
        
        Parameters:
        -----------
        demand_stats : Dict
            Demand statistics
        safety_stock : float
            Safety stock quantity
            
        Returns:
        --------
        float
            Reorder point quantity
        """
        if demand_stats is None:
            return 0
            
        # Reorder Point = Expected demand during lead time + Safety stock
        reorder_point = demand_stats['lead_time_demand_mean'] + safety_stock
        
        return max(0, reorder_point)
    
    def calculate_order_quantity(self, current_inventory: float, 
                               reorder_point: float, 
                               safety_stock: float,
                               max_stock_multiplier: float = 1.5) -> float:
        """
        Calculate recommended order quantity.
        
        Parameters:
        -----------
        current_inventory : float
            Current inventory level
        reorder_point : float
            Reorder point
        safety_stock : float
            Safety stock level
        max_stock_multiplier : float
            Multiplier for maximum stock level
            
        Returns:
        --------
        float
            Recommended order quantity
        """
        # Simple max stock rule: reorder point + additional safety buffer
        max_stock = reorder_point + (safety_stock * max_stock_multiplier)
        
        # Order quantity to reach max stock
        order_quantity = max(0, max_stock - current_inventory)
        
        return order_quantity
    
    def simulate_inventory_policy(self, demand_stats: Dict, 
                                reorder_point: float,
                                safety_stock: float,
                                simulation_days: int = 90) -> Dict:
        """
        Simulate inventory policy performance.
        
        Parameters:
        -----------
        demand_stats : Dict
            Demand statistics
        reorder_point : float
            Reorder point
        safety_stock : float
            Safety stock
        simulation_days : int
            Number of days to simulate
            
        Returns:
        --------
        Dict
            Simulation results
        """
        if demand_stats is None:
            return {'stockouts': 0, 'service_level': 0, 'avg_inventory': 0}
        
        # Simulate daily demand
        daily_demands = np.random.normal(
            demand_stats['daily_demand_mean'],
            demand_stats['daily_demand_std'],
            simulation_days
        )
        daily_demands = np.maximum(0, daily_demands)  # No negative demand
        
        # Simulate inventory levels
        inventory = reorder_point + safety_stock  # Start with max stock
        stockouts = 0
        inventory_levels = []
        
        for demand in daily_demands:
            # Check if we can meet demand
            if inventory < demand:
                stockouts += 1
                inventory = 0  # Stockout
            else:
                inventory -= demand
            
            # Reorder if below reorder point (assuming instant replenishment)
            if inventory <= reorder_point:
                inventory = reorder_point + safety_stock
            
            inventory_levels.append(inventory)
        
        service_level = 1 - (stockouts / simulation_days)
        avg_inventory = np.mean(inventory_levels)
        
        return {
            'stockouts': stockouts,
            'service_level': service_level,
            'avg_inventory': avg_inventory,
            'simulation_days': simulation_days
        }
    
    def optimize_inventory(self, historical_data: pd.DataFrame, 
                         current_inventory: Optional[Dict] = None,
                         service_level: float = 0.95, 
                         lead_time_days: int = 7,
                         run_simulation: bool = False) -> pd.DataFrame:
        """
        Optimize inventory for all store-family combinations.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical sales data
        current_inventory : Dict, optional
            Current inventory levels by (store, family)
        service_level : float
            Target service level (0-1)
        lead_time_days : int
            Lead time in days
        run_simulation : bool
            Whether to run policy simulation
            
        Returns:
        --------
        pd.DataFrame
            Inventory optimization results
        """
        print(f"Optimizing inventory for service level {service_level*100}% and lead time {lead_time_days} days...")
        
        results = []
        
        store_families = historical_data[['store_nbr', 'family']].drop_duplicates()
        
        for _, row in store_families.iterrows():
            store = row['store_nbr']
            family = row['family']
            
            # Calculate demand statistics
            demand_stats = self.calculate_demand_stats(
                historical_data, store, family, lead_time_days
            )
            
            if demand_stats is None:
                continue
                
            # Store demand stats for later use
            self.demand_stats[(store, family)] = demand_stats
            
            # Calculate safety stock
            safety_stock = self.calculate_safety_stock(demand_stats, service_level)
            
            # Calculate reorder point
            reorder_point = self.calculate_reorder_point(demand_stats, safety_stock)
            
            # Get or simulate current inventory
            if current_inventory is None:
                # Simulate current inventory (normally distributed around 80% of reorder point)
                current_stock = max(0, np.random.normal(
                    reorder_point * 0.8, 
                    reorder_point * 0.2
                ))
            else:
                current_stock = current_inventory.get((store, family), reorder_point * 0.8)
            
            # Calculate order quantity
            order_quantity = self.calculate_order_quantity(
                current_stock, reorder_point, safety_stock
            )
            
            # Determine if reorder is needed
            needs_reorder = current_stock <= reorder_point
            
            # Calculate max stock level
            max_stock = reorder_point + safety_stock * 0.5
            
            result = {
                'store_nbr': store,
                'family': family,
                'daily_demand_mean': demand_stats['daily_demand_mean'],
                'daily_demand_std': demand_stats['daily_demand_std'],
                'lead_time_days': lead_time_days,
                'service_level': service_level,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'current_inventory': current_stock,
                'max_stock': max_stock,
                'order_quantity': order_quantity,
                'needs_reorder': needs_reorder,
                'data_points': demand_stats['data_points']
            }
            
            # Run simulation if requested
            if run_simulation:
                sim_results = self.simulate_inventory_policy(
                    demand_stats, reorder_point, safety_stock
                )
                result.update({
                    'simulated_service_level': sim_results['service_level'],
                    'simulated_stockouts': sim_results['stockouts'],
                    'avg_inventory_level': sim_results['avg_inventory']
                })
            
            results.append(result)
        
        self.results = pd.DataFrame(results)
        print(f"Optimization completed for {len(self.results)} store-family combinations")
        return self.results
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary statistics of optimization results.
        
        Returns:
        --------
        Dict
            Summary statistics
        """
        if self.results is None or len(self.results) == 0:
            return {}
        
        return {
            'total_items': len(self.results),
            'items_needing_reorder': self.results['needs_reorder'].sum(),
            'reorder_percentage': (self.results['needs_reorder'].sum() / len(self.results)) * 100,
            'total_safety_stock': self.results['safety_stock'].sum(),
            'total_order_quantity': self.results['order_quantity'].sum(),
            'avg_service_level': self.results['service_level'].mean(),
            'avg_reorder_point': self.results['reorder_point'].mean(),
            'avg_safety_stock': self.results['safety_stock'].mean(),
            'total_current_inventory': self.results['current_inventory'].sum()
        }
    
    def export_results(self, filename: str = 'inventory_optimization_results.csv') -> None:
        """
        Export optimization results to CSV.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        if self.results is not None:
            self.results.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            print("No results to export. Run optimize_inventory() first.")


class ABCAnalysis:
    """
    ABC Analysis for inventory classification.
    
    Classifies items based on their importance (typically by revenue or volume).
    """
    
    def __init__(self):
        self.results = None
    
    def classify_items(self, data: pd.DataFrame, 
                      value_col: str = 'sales',
                      groupby_cols: List[str] = ['store_nbr', 'family'],
                      a_threshold: float = 0.8,
                      b_threshold: float = 0.95) -> pd.DataFrame:
        """
        Perform ABC classification.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        value_col : str
            Column to use for classification
        groupby_cols : list
            Columns to group by
        a_threshold : float
            Threshold for A items (cumulative percentage)
        b_threshold : float
            Threshold for B items (cumulative percentage)
            
        Returns:
        --------
        pd.DataFrame
            Data with ABC classification
        """
        print(f"Performing ABC analysis on {len(data)} records...")
        
        # Calculate total value by group
        summary = data.groupby(groupby_cols)[value_col].sum().reset_index()
        
        # Sort by value descending
        summary = summary.sort_values(value_col, ascending=False)
        
        # Calculate cumulative percentage
        summary['cumulative_value'] = summary[value_col].cumsum()
        summary['cumulative_percentage'] = summary['cumulative_value'] / summary[value_col].sum()
        summary['percentage_of_total'] = (summary[value_col] / summary[value_col].sum()) * 100
        
        # Assign ABC classification
        summary['abc_class'] = 'C'
        summary.loc[summary['cumulative_percentage'] <= a_threshold, 'abc_class'] = 'A'
        summary.loc[
            (summary['cumulative_percentage'] > a_threshold) & 
            (summary['cumulative_percentage'] <= b_threshold), 
            'abc_class'
        ] = 'B'
        
        self.results = summary
        print(f"ABC Analysis completed:")
        print(f"- A items: {len(summary[summary['abc_class'] == 'A'])} ({(len(summary[summary['abc_class'] == 'A']) / len(summary) * 100):.1f}%)")
        print(f"- B items: {len(summary[summary['abc_class'] == 'B'])} ({(len(summary[summary['abc_class'] == 'B']) / len(summary) * 100):.1f}%)")
        print(f"- C items: {len(summary[summary['abc_class'] == 'C'])} ({(len(summary[summary['abc_class'] == 'C']) / len(summary) * 100):.1f}%)")
        
        return summary
    
    def get_abc_summary(self) -> Dict:
        """Get ABC analysis summary."""
        if self.results is None:
            return {}
        
        summary_stats = {}
        for class_type in ['A', 'B', 'C']:
            class_data = self.results[self.results['abc_class'] == class_type]
            summary_stats[f'Class_{class_type}'] = {
                'count': len(class_data),
                'total_value': class_data['sales'].sum(),
                'avg_value': class_data['sales'].mean(),
                'percentage_of_items': (len(class_data) / len(self.results)) * 100,
                'percentage_of_value': (class_data['sales'].sum() / self.results['sales'].sum()) * 100
            }
        
        return summary_stats
    
    def export_results(self, filename: str = 'abc_analysis_results.csv') -> None:
        """Export ABC analysis results to CSV."""
        if self.results is not None:
            self.results.to_csv(filename, index=False)
            print(f"ABC analysis results exported to {filename}")
        else:
            print("No results to export. Run classify_items() first.")


def calculate_economic_order_quantity(annual_demand: float, 
                                    ordering_cost: float, 
                                    holding_cost_per_unit: float) -> Dict:
    """
    Calculate Economic Order Quantity (EOQ).
    
    Parameters:
    -----------
    annual_demand : float
        Annual demand in units
    ordering_cost : float
        Fixed cost per order
    holding_cost_per_unit : float
        Annual holding cost per unit
        
    Returns:
    --------
    Dict
        EOQ calculations
    """
    if holding_cost_per_unit <= 0:
        raise ValueError("Holding cost must be positive")
    
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    
    # Calculate associated costs
    annual_ordering_cost = (annual_demand / eoq) * ordering_cost
    annual_holding_cost = (eoq / 2) * holding_cost_per_unit
    total_annual_cost = annual_ordering_cost + annual_holding_cost
    
    return {
        'eoq': eoq,
        'annual_ordering_cost': annual_ordering_cost,
        'annual_holding_cost': annual_holding_cost,
        'total_annual_cost': total_annual_cost,
        'order_frequency': annual_demand / eoq,
        'days_between_orders': 365 / (annual_demand / eoq)
    }