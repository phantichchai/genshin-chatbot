import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List

class WebSearch:

    def __init__(
            self,
            search_url: str='www.google.com',
            excluded_domains: List[str]=['google.com', 'youtube.com']
        ) -> None:
        self.result_urls = []
        self.excluded_domains = excluded_domains
        self.search_url = search_url
    

    def search(self, query):
        url = f"https://{self.search_url}/search?hl=en&q={query}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')
            
            urls = set()

            for link in links:
                href = link.get('href')
                if href and href.startswith("https://"):
                    urls.add(href)
        else:
            print(f"Failed to retrieve search results. Status code: {response.status_code}")

        return self.filter_urls_not_containing_domains(urls, self.excluded_domains)


    
    def filter_urls_not_containing_domains(self, urls, excluded_domains):
        """
        Filter a list of URLs to exclude those containing any of the specified domains.

        Parameters:
        - urls (list): A list of URLs.
        - excluded_domains (list): A list of domains to exclude.

        Returns:
        - list: A filtered list of URLs.
        """
        for url in urls:
            parsed_url = urlparse(url)
            if all(domain not in parsed_url.netloc for domain in excluded_domains):
                self.result_urls.append(url)

        return self.result_urls
    
if __name__ == "__main__":
    websearch = WebSearch()
    result_url = websearch.search('Furina')
    print(result_url)