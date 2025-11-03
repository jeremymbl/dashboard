#!/usr/bin/env python3
"""
Helpers for diagnosticians analysis and visualization.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List
from scripts.diag_list import DiagListRessource
from scripts.diag_list_clean import clean_registry


def analyze_certificates_from_df(df: pd.DataFrame, csv_date: datetime = None) -> Dict:
    """
    Analyze a cleaned diagnosticians DataFrame to count all certificate types.
    Only counts certificates that are valid at the CSV publication date.

    Args:
        df: Cleaned DataFrame with columns for all certification types (*_debut, *_fin)
        csv_date: Date of CSV publication to check certificate validity against

    Returns:
        Dictionary with counts and percentages: {
            'counts': {'DPE': count, 'Audit': count, 'Amiante': count, 'Plomb': count, 'Autres': count},
            'total': total_certifications,
            'percentages': {'DPE': %, 'Audit': %, 'Amiante': %, 'Plomb': %, 'Autres': %}
        }
    """
    # Helper function to count valid certificates
    def count_valid_certs(prefix: str) -> int:
        debut_col = f'{prefix}_debut'
        fin_col = f'{prefix}_fin'

        if debut_col not in df.columns or fin_col not in df.columns:
            return 0

        if csv_date is None:
            # Count all certificates
            return df[df[debut_col].notna() & df[fin_col].notna()].shape[0]
        else:
            # Convert csv_date to date if it's a datetime object
            check_date = csv_date.date() if hasattr(csv_date, 'date') else csv_date
            # Count only valid certificates at csv_date
            mask = (
                df[debut_col].notna() &
                df[fin_col].notna() &
                (df[fin_col].dt.date >= check_date)
            )
            return df[mask].shape[0]

    # Count each certification type
    dpe_count = count_valid_certs('DPE')
    audit_count = count_valid_certs('Audit')
    amiante_count = count_valid_certs('Amiante')
    plomb_count = count_valid_certs('Plomb')

    # "Autres" includes: Electricité, Gaz, Termites, DRIPP
    electricite_count = count_valid_certs('Electricite')
    gaz_count = count_valid_certs('Gaz')
    termites_count = count_valid_certs('Termites')
    dripp_count = count_valid_certs('DRIPP')
    autres_count = electricite_count + gaz_count + termites_count + dripp_count

    # Calculate totals
    total_certifications = dpe_count + audit_count + amiante_count + plomb_count + autres_count

    # Calculate percentages
    def calc_percentage(count: int, total: int) -> float:
        return round((count / total * 100), 2) if total > 0 else 0.0

    counts = {
        'DPE': dpe_count,
        'Audit': audit_count,
        'Amiante': amiante_count,
        'Plomb': plomb_count,
        'Autres': autres_count
    }

    percentages = {
        'DPE': calc_percentage(dpe_count, total_certifications),
        'Audit': calc_percentage(audit_count, total_certifications),
        'Amiante': calc_percentage(amiante_count, total_certifications),
        'Plomb': calc_percentage(plomb_count, total_certifications),
        'Autres': calc_percentage(autres_count, total_certifications)
    }

    return {
        'counts': counts,
        'total': total_certifications,
        'percentages': percentages,
        'total_diagnosticians': len(df)
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
                'counts': {'DPE': X, 'Audit': Y, 'Amiante': Z, 'Plomb': W, 'Autres': V},
                'total': total_certifications,
                'percentages': {'DPE': X%, 'Audit': Y%, 'Amiante': Z%, 'Plomb': W%, 'Autres': V%},
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
            cert_data = analyze_certificates_from_df(cleaned_df, resource.date)

            # Format month string
            month_str = resource.date.strftime('%Y-%m')

            results.append({
                'month': month_str,
                'date': resource.date,
                'counts': cert_data['counts'],
                'total': cert_data['total'],
                'percentages': cert_data['percentages'],
                'total_diagnosticians': cert_data['total_diagnosticians'],
                'resource_label': resource.label
            })

        except Exception as e:
            print(f"Error processing {resource.label}: {e}")
            # Add empty data for this month to maintain continuity
            month_str = resource.date.strftime('%Y-%m') if resource.date else 'Unknown'
            results.append({
                'month': month_str,
                'date': resource.date,
                'counts': {'DPE': 0, 'Audit': 0, 'Amiante': 0, 'Plomb': 0, 'Autres': 0},
                'total': 0,
                'percentages': {'DPE': 0, 'Audit': 0, 'Amiante': 0, 'Plomb': 0, 'Autres': 0},
                'total_diagnosticians': 0,
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
    Prepare data for Plotly chart display with percentages.

    Args:
        monthly_data: List of monthly analysis results with counts and percentages

    Returns:
        DataFrame ready for plotting with columns: month, certificate_type, percentage, count, total
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

    # Certificate types in display order
    cert_types = ['DPE', 'Audit', 'Amiante', 'Plomb', 'Autres']
    cert_labels = {
        'DPE': 'DPE',
        'Audit': 'Audit énergétique',
        'Amiante': 'Amiante',
        'Plomb': 'Plomb',
        'Autres': 'Autres (Élec, Gaz, Termites, etc.)'
    }

    for data in reversed_data:
        # Convertir YYYY-MM en label français
        year, month = data['month'].split('-')
        month_label = f"{month_names[int(month)]} {year}"

        # Add data for each certificate type
        for cert_type in cert_types:
            chart_data.append({
                'month': month_label,
                'certificate_type': cert_labels[cert_type],
                'percentage': data['percentages'].get(cert_type, 0),
                'count': data['counts'].get(cert_type, 0),
                'total': data['total']
            })

    # Create DataFrame
    df = pd.DataFrame(chart_data)

    # Debug: Print the final DataFrame
    print("\n===== FINAL CHART DATA =====")
    print(df.head(20))

    return df
