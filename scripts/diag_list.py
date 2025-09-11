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
        """Fetch all available diagnostic resources from data.gouv.fr.

        Returns:
            List of resources sorted by date (most recent first)
        """

        def _parse_date_from_label(label: str) -> Optional[Date]:
            """Parse date from French diagnostic resource labels.

            Supports three formats:
            1. YYYYMMDD-filename.csv (e.g., "20250701-annuaire-diagnostiqueurs.csv")
            2. "Maj du DD/MM/YYYY" (e.g., "Annuaire (Maj du 31/12/2019)")
            3. "dernière mise à jour" (returns None - indicates latest)

            Args:
                label: The resource label text to parse

            Returns:
                Parsed date or None if no specific date found
            """
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

            # Format 3: "dernière mise à jour" or no specific date
            return None

        async def _fetch_html_with_playwright(url: str, timeout: int = 30000) -> Optional[str]:
            """Fetch rendered HTML content using Playwright for JavaScript-heavy sites.

            Args:
                url: The URL to fetch
                timeout: Request timeout in milliseconds

            Returns:
                Rendered HTML content or None if failed
            """
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()

                    await page.set_extra_http_headers(
                        {"User-Agent": "Mozilla/5.0 (compatible; ScraperBot/1.0; +https://example.com/bot)"}
                    )

                    await page.goto(url, timeout=timeout)
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)  # Wait for Vue app to render

                    html = await page.content()
                    await browser.close()
                    return html

            except Exception as e:
                print(Panel(f"[red]Playwright request failed:[/red] {e}", title="Error", style="red"))
                return None

        def _scrap_diag_list_ressources(html: str) -> list[DiagListRessource]:
            """Extract and parse diagnostic resources from HTML content.

            Args:
                html: HTML content containing resource options

            Returns:
                List of resources sorted by date (None date first, then descending)

            Raises:
                ValueError: If there's not exactly one undated resource at the top
            """
            soup = BeautifulSoup(html, "html.parser")
            select = soup.find("select")

            if not select or not isinstance(select, Tag):
                print(Panel("[yellow]No <select> element found.[/yellow]", title="Notice", style="yellow"))
                return []

            files: list[DiagListRessource] = []
            for option in select.find_all("option"):
                if isinstance(option, Tag):
                    value = option.get("value", "")
                    text = option.get_text(strip=True)

                    if value and text:  # Skip empty options
                        parsed_date = _parse_date_from_label(text)
                        files.append(
                            DiagListRessource(
                                id=str(value),
                                label=str(text),
                                date=parsed_date,
                                index=-1,  # Set after sorting
                            )
                        )

            # Separate dated and undated resources
            dated = [f for f in files if f.date is not None]
            undated = [f for f in files if f.date is None]

            # Warn if not exactly one undated resource
            if len(undated) != 1:
                print(Panel(f"[yellow]WARNING: Expected exactly one undated resource (latest), but found {len(undated)}.\nCheck the data source for anomalies.[/yellow]", title="Resource List Warning", style="yellow"))

            # Sort dated resources by date descending (most recent first)
            dated.sort(key=lambda x: (0, x.date) if x.date is not None else (1, Date.min), reverse=True)

            # Combine: undated first, then dated
            files = undated + dated

            # Set index positions
            for idx, f in enumerate(files):
                f.index = idx

            return files

        url = "https://explore.data.gouv.fr/fr/resources/a0826a82-df34-4455-b0fc-37b9516554c5"
        html = await _fetch_html_with_playwright(url)
        if not html:
            return []
        return _scrap_diag_list_ressources(html)

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
