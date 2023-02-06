from googleapiclient.discovery import build
import urllib.request

def search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res
    
def download(url):
    req = urllib.request.Request(
        url=url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
    )
    content = urllib.request.urlopen(req).read()
    return content 