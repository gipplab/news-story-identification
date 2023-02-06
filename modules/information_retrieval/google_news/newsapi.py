import urllib.request
from newsapi import NewsApiClient

def search(search_term, api_key, **kwargs):
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=search_term,language='en', page=1)
    return all_articles
    
def download(url):
    # opener = urllib.request.FancyURLopener({})
    # f = opener.open(url)
    # content = f.read()

    req = urllib.request.Request(
        url=url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    content = urllib.request.urlopen(req).read()
    return content 