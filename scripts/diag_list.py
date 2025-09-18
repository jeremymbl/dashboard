#!/usr/bin/env python3
"""
Diagnosticians Registry Scraper for data.gouv.fr

Scrapes the French diagnosticians registry from data.gouv.fr and provides
structured access to CSV files with pandas integration.

Installation:
    uv add beautifulsoup4 pandas httpx playwright
    uv run playwright install chromium

Usage:
    # Fetch all resources
    resources = DiagListRessource.fetch_all_sync()

    # Get DataFrame for most recent
    df = asyncio.run(resources[0].fetch_csv_dataframe())

    # Or by ID directly
    df = await DiagListRessource.fetch_csv_dataframe_by_id("resource-id")

Features:
    - Automatic French date parsing (3 formats)
    - Sorted by date (most recent first)
    - Both sync and async APIs
    - Direct pandas DataFrame integration
"""

import asyncio
import re
from datetime import date as Date
from io import BytesIO
from typing import Any, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from rich import print
from rich.panel import Panel
from rich.table import Table


class DiagListRessource(BaseModel):
    """A diagnostic resource from the French diagnosticians registry.

    Attributes:
        index: Position in the sorted list (0-based)
        id: Unique resource identifier from data.gouv.fr
        label: Human-readable resource name
        date: Parsed date from label (None for latest/undated)
    """

    index: int = Field(..., description="Position in the sorted list")
    id: str = Field(..., description="Unique resource identifier")
    label: str = Field(..., description="Human-readable resource name")
    date: Optional[Date] = Field(None, description="Parsed date from label")

    @property
    def page_url(self) -> str:
        """Get the page URL for this resource."""
        return f"https://explore.data.gouv.fr/fr/datasets/5bc5df57634f417a900a5ed0/#/resources/{self.id}"

    @property
    def csv_url(self) -> str:
        """Get the direct download URL for this resource."""
        return f"https://www.data.gouv.fr/fr/datasets/r/{self.id}"

    @staticmethod
    async def fetch_csv_data_by_id(resource_id: str) -> bytes:
        """Download CSV file as bytes for a given resource ID."""
        url = f"https://www.data.gouv.fr/fr/datasets/r/{resource_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content

    @staticmethod
    async def fetch_csv_dataframe_by_id(resource_id: str, sep: str = ";", **kwargs: Any) -> pd.DataFrame:
        """Download CSV and return as pandas DataFrame for a given resource ID.

        Args:
            resource_id: The data.gouv.fr resource identifier
            sep: Column separator (default ';' for French CSVs)
            **kwargs: Additional arguments passed to pandas.read_csv

        Returns:
            pandas DataFrame containing the CSV data
        """
        csv_bytes = await DiagListRessource.fetch_csv_data_by_id(resource_id)
        return pd.read_csv(BytesIO(csv_bytes), sep=sep, **kwargs)

    async def fetch_csv_data(self) -> bytes:
        """Download this resource's CSV file as bytes."""
        return await self.fetch_csv_data_by_id(self.id)

    async def fetch_csv_dataframe(self, sep: str = ";", **kwargs: Any) -> pd.DataFrame:
        """Download this resource's CSV as pandas DataFrame.

        Args:
            sep: Column separator (default ';' for French CSVs)
            **kwargs: Additional arguments passed to pandas.read_csv

        Returns:
            pandas DataFrame containing the CSV data
        """
        return await self.fetch_csv_dataframe_by_id(self.id, sep=sep, **kwargs)

    @staticmethod
    async def fetch_all() -> list["DiagListRessource"]:
        """Fetch all available diagnostic resources from data.gouv.fr using the official API."""
        
        dataset_id = "5bc5df57634f417a900a5ed0"  # ID de l'annuaire des diagnostiqueurs
        api_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(api_url, timeout=20.0)
                response.raise_for_status()
                data = response.json()
            except httpx.RequestError as e:
                print(f"API request failed: {e}")
                return []

        # La fonction de parsing de date est toujours utile
        def _parse_date_from_label(label: str) -> Optional[Date]:
            """Parse date from French diagnostic resource labels."""
            # Format 1: YYYYMMDD-filename.csv
            format1_pattern = r"^(\d{4})(\d{2})(\d{2})-"
            match = re.match(format1_pattern, label)
            if match:
                year, month, day = match.groups()
                try:
                    return Date(int(year), int(month), int(day))
                except ValueError:
                    pass

            # Format 2: "Maj du DD/MM/YYYY"
            format2_pattern = r"Maj du (\d{1,2})/(\d{1,2})/(\d{4})"
            match = re.search(format2_pattern, label)
            if match:
                day, month, year = match.groups()
                try:
                    return Date(int(year), int(month), int(day))
                except ValueError:
                    pass
            return None

        files: list[DiagListRessource] = []
        for resource in data.get("resources", []):
            if resource.get("format", "").lower() == "csv":
                resource_id = resource.get("id", "")
                title = resource.get("title", "")
                
                if resource_id and title:
                    parsed_date = _parse_date_from_label(title)
                    files.append(
                        DiagListRessource(
                            id=resource_id,
                            label=title,
                            date=parsed_date,
                            index=-1,  # Sera défini après le tri
                        )
                    )

        # La logique de tri reste la même
        dated = sorted([f for f in files if f.date is not None], key=lambda x: x.date, reverse=True)
        undated = [f for f in files if f.date is None]
        
        # Le fichier le plus récent sans date est souvent le "dernier", on le met en premier
        files = undated + dated
        
        for idx, f in enumerate(files):
            f.index = idx
            
        return files

    @classmethod
    def fetch_all_sync(cls) -> list["DiagListRessource"]:
        """Synchronously fetch all available diagnostic resources.

        Returns:
            List of resources sorted by date (most recent first)
        """
        return asyncio.run(cls.fetch_all())

    @classmethod
    def fetch_last_12_months_sync(cls) -> list["DiagListRessource"]:
        """Fetch the 12 most recent monthly diagnostic resources.
        
        Returns:
            List of resources for the last 12 months (most recent first)
        """
        all_resources = cls.fetch_all_sync()
        
        # Filter to get only dated resources (exclude "latest")
        dated_resources = [r for r in all_resources if r.date is not None]
        
        # Group by month and take the oldest (first of month) for each month
        monthly_resources = {}
        for resource in dated_resources:
            month_key = (resource.date.year, resource.date.month)
            if month_key not in monthly_resources:
                monthly_resources[month_key] = resource
            else:
                # Keep the oldest date for each month (closest to first of month)
                if resource.date < monthly_resources[month_key].date:
                    monthly_resources[month_key] = resource
        
        # Sort by date descending and take last 12 months
        sorted_monthly = sorted(monthly_resources.values(), key=lambda x: x.date, reverse=True)
        return sorted_monthly[:12]


def print_diag_list_ressources(files: list[DiagListRessource]) -> None:
    """Display diagnostic resources in a formatted table.

    Args:
        files: List of diagnostic resources to display
    """
    if not files:
        print(Panel("[yellow]No diagnostic files found.[/yellow]", title="Notice", style="yellow"))
        return

    table = Table(title="Diagnostic Files", show_lines=True)
    table.add_column("#", style="bold magenta")
    table.add_column("Label", style="bold cyan")
    table.add_column("ID", style="green")
    table.add_column("Date", style="yellow")

    for file in files:
        date_str = file.date.strftime("%d/%m/%Y") if file.date else "Latest"
        clickable_label = f"[link={file.page_url}]{file.label}[/link]"
        table.add_row(str(file.index), clickable_label, file.id, date_str)

    print(table)


def render_html_panel(html: str, url: str) -> None:
    """Render HTML content in a rich panel for debugging.

    Args:
        html: HTML content to display
        url: Source URL for subtitle
    """
    truncated = html[:1000] + "\n...[truncated]" if len(html) > 1000 else html
    print(Panel(truncated, title="Raw HTML", subtitle=url, border_style="blue", expand=False))


if __name__ == "__main__":
    # Fetch and display all resources
    resources = DiagListRessource.fetch_all_sync()
    print_diag_list_ressources(resources)

    # Print summary statistics
    print(f"\n[bold green]Found {len(resources)} diagnostic files[/bold green]")
    files_with_dates = sum(1 for resource in resources if resource.date is not None)
    print(f"[bold blue]{files_with_dates} files have parsed dates[/bold blue]")

    # Show preview of most recent resource
    if resources:
        print("\n[bold]Preview of most recent resource:[/bold]")
        df = asyncio.run(resources[0].fetch_csv_dataframe())
        print(df.head())
