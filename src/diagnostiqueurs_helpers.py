#!/usr/bin/env python3
"""
Helpers for diagnosticians analysis and visualization.
"""

import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from scripts.diag_list import DiagListRessource
from scripts.diag_list_clean import clean_registry


def analyze_certificates_from_df(df: pd.DataFrame, csv_date: datetime = None) -> Dict[str, int]:
    """
    Analyze a cleaned diagnosticians DataFrame to count certificate types.
    Only counts certificates that are valid at the CSV publication date.
    
    Args:
        df: Cleaned DataFrame with columns 'DPE_debut', 'DPE_fin', 'Audit_debut', 'Audit_fin'
        csv_date: Date of CSV publication to check certificate validity against
        
    Returns:
        Dictionary with counts: {'DPE': count, 'Audit': count}
    """
    # If no csv_date provided, count all certificates (backward compatibility)
    if csv_date is None:
        # Count DPE certificates (those with both DPE_debut and DPE_fin filled)
        dpe_count = df[
            df['DPE_debut'].notna() & 
            df['DPE_fin'].notna()
        ].shape[0]
        
        # Count Audit certificates (those with both Audit_debut and Audit_fin filled)
        audit_count = df[
            df['Audit_debut'].notna() & 
            df['Audit_fin'].notna()
        ].shape[0]
    else:
        # Convert csv_date to date if it's a datetime object
        if hasattr(csv_date, 'date'):
            csv_date = csv_date.date()
        
        # Count DPE certificates that are valid at csv_date
        dpe_mask = (
            df['DPE_debut'].notna() & 
            df['DPE_fin'].notna() &
            (df['DPE_fin'].dt.date >= csv_date)
        )
        dpe_count = df[dpe_mask].shape[0]
        
        # Count Audit certificates that are valid at csv_date
        audit_mask = (
            df['Audit_debut'].notna() & 
            df['Audit_fin'].notna() &
            (df['Audit_fin'].dt.date >= csv_date)
        )
        audit_count = df[audit_mask].shape[0]
    
    return {
        'DPE': dpe_count,
        'Audit': audit_count
    }


def fetch_and_analyze_monthly_data() -> List[Dict]:
    """
    Fetch the last 12 months of diagnosticians data and analyze certificates.
    
    Returns:
        List of dictionaries with monthly data:
        [
            {
                'month': '2024-01',
                'date': datetime_object,
                'DPE': count,
                'Audit': count,
                'resource_label': 'CSV filename'
            },
            ...
        ]
    """
    # Get the 12 monthly resources
    monthly_resources = DiagListRessource.fetch_last_12_months_sync()
    
    # Debug: Print the resources we're working with
    print("\n===== RESOURCES FETCHED =====")
    for i, resource in enumerate(monthly_resources):
        print(f"{i+1}. {resource.label} - Date: {resource.date}")
    
    results = []
    
    for resource in monthly_resources:
        try:
            # Use a new event loop for each resource to avoid nested asyncio.run() calls
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Download and clean the CSV
            raw_df = loop.run_until_complete(resource.fetch_csv_dataframe())
            loop.close()
            
            cleaned_df = clean_registry(raw_df)
            
            # Analyze certificates with CSV date validation
            cert_counts = analyze_certificates_from_df(cleaned_df, resource.date)
            
            # Format month string
            month_str = resource.date.strftime('%Y-%m')
            
            results.append({
                'month': month_str,
                'date': resource.date,
                'DPE': cert_counts['DPE'],
                'Audit': cert_counts['Audit'],
                'resource_label': resource.label
            })
            
        except Exception as e:
            print(f"Error processing {resource.label}: {e}")
            # Add empty data for this month to maintain continuity
            month_str = resource.date.strftime('%Y-%m') if resource.date else 'Unknown'
            results.append({
                'month': month_str,
                'date': resource.date,
                'DPE': 0,
                'Audit': 0,
                'resource_label': resource.label,
                'error': str(e)
            })
    
    # Sort by date (most recent first, but we'll reverse for chart display)
    results.sort(key=lambda x: x['date'] if x['date'] else datetime.min, reverse=True)
    
    # Debug: Print the results after sorting
    print("\n===== RESULTS AFTER SORTING =====")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['month']} - {result['resource_label']}")
    
    # Filter to keep only months between September 2025 and December 2024
    # Define the date range
    start_date = datetime(2024, 12, 1).date()  # December 2024
    end_date = datetime(2025, 9, 30).date()    # September 2025
    
    # Filter results to keep only those within the date range
    filtered_results = [r for r in results if start_date <= r['date'] <= end_date]
    
    # Group by month and keep the oldest entry for each month
    monthly_results = {}
    for result in filtered_results:
        month_key = result['month']
        if month_key not in monthly_results:
            monthly_results[month_key] = result
        elif result['date'] < monthly_results[month_key]['date']:
            # Keep the oldest date for each month (closest to first of month)
            monthly_results[month_key] = result
    
    # Convert back to list and sort by date (most recent first)
    final_results = list(monthly_results.values())
    final_results.sort(key=lambda x: x['date'], reverse=True)
    
    # Debug: Print the final results
    print("\n===== FINAL RESULTS (FILTERED BY DATE RANGE) =====")
    for i, result in enumerate(final_results):
        print(f"{i+1}. {result['month']} - {result['resource_label']}")
    
    return final_results


def prepare_chart_data(monthly_data: List[Dict]) -> pd.DataFrame:
    """
    Prepare data for Plotly chart display.
    
    Args:
        monthly_data: List of monthly analysis results
        
    Returns:
        DataFrame ready for plotting with columns: month, certificate_type, count
    """
    # Debug: Print the input data
    print("\n===== INPUT DATA FOR CHART =====")
    for i, data in enumerate(monthly_data):
        print(f"{i+1}. {data['month']} - {data['resource_label']}")
    
    chart_data = []
    
    # Reverse to show oldest to newest (left to right on chart)
    reversed_data = list(reversed(monthly_data))
    
    # Debug: Print the reversed data
    print("\n===== REVERSED DATA FOR CHART (OLDEST TO NEWEST) =====")
    for i, data in enumerate(reversed_data):
        print(f"{i+1}. {data['month']} - {data['resource_label']}")
    
    # Créer les labels de mois en français
    month_names = {
        1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
        7: "Juil", 8: "Août", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
    }
    
    for data in reversed_data:
        # Convertir YYYY-MM en label français
        year, month = data['month'].split('-')
        month_label = f"{month_names[int(month)]} {year}"
        
        chart_data.append({
            'month': month_label,
            'certificate_type': 'DPE',
            'count': data['DPE']
        })
        chart_data.append({
            'month': month_label, 
            'certificate_type': 'Audit énergétique',
            'count': data['Audit']
        })
    
    # Create DataFrame
    df = pd.DataFrame(chart_data)
    
    # Debug: Print the final DataFrame
    print("\n===== FINAL CHART DATA =====")
    print(df)
    
    return df
