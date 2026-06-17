"""
utils/yf_session.py
--------------------
Shared yfinance session using curl_cffi with Chrome impersonation.
Fixes YFTzMissingError caused by Yahoo Finance blocking default
requests/urllib3 user-agents. Imported by fetcher.py and sentiment.py.
"""

from curl_cffi import requests as curl_requests

YF_SESSION = curl_requests.Session(impersonate="chrome")