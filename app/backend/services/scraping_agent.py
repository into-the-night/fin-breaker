# Scraping Agent
# Handles crawling of filings and documents

from fastapi import APIRouter, Query
from bs4 import BeautifulSoup
import requests

router = APIRouter(prefix="/scraping", tags=["Scraping Agent"])

@router.get("/filing")
def get_filing(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. TSM"),
    doc_type: str = Query("10-K", description="Filing type, e.g. 10-K, 20-F, 6-K")
):
    # Simple EDGAR search for US stocks, fallback to Yahoo for others
    base_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={doc_type}&dateb=&owner=exclude&count=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(base_url, headers=headers)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, "html.parser")
        doc_link = soup.find('a', {'id': 'documentsbutton'})
        if doc_link:
            return {"filing_url": f"https://www.sec.gov{doc_link['href']}"}
        return {"message": "No filing found"}
    return {"error": "Failed to fetch filing"}

@router.get("/")
def root():
    return {"status": "Scraping Agent running"}
