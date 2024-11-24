import requests
from flask import Flask
from bs4 import BeautifulSoup
from db_schema import db, Company
import time

tickers = []
response = requests.get("https://stockanalysis.com/list/biggest-companies/")
soup = BeautifulSoup(response.text, 'html.parser')
main_table = soup.find('table', {'id': 'main-table'})
all_tickers = [[a_tag.get('href').replace("/stocks/", "")[:-1].upper()] for a_tag in main_table.find_all('a')][:300]

# Converts the necessary information obtained from AlphaVantage into the database format
def get_ticker_info(n):
  for ticker in all_tickers[n:n+50]: 
    overview = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker[0].replace('.', '-')}&apikey=QCLLY08TISZBMXL9").json()
    db.session.add(Company(ticker[0], overview["Name"], overview["Sector"].title() ))
    time.sleep(2)

for i in range(0, 300, 50):
  get_ticker_info(i)

print(all_tickers)