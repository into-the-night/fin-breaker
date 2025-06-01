# Filings Loader
# Loads filings and documents for analysis

import requests
from bs4 import BeautifulSoup

class FilingsLoader:
    def get_filing(self, ticker, doc_type="10-K"):
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
