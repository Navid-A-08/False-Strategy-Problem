"""Downloads daily S&P 500 index closes (Yahoo Finance ^GSPC, 1990-01-01 to 2026-09-29) to
results/sp500_daily.csv. The file used in the paper is already included; run this only to refresh it.
Note: Yahoo data can be revised, so a fresh download may differ slightly."""
import yfinance as yf
d = yf.download('^GSPC', start='1990-01-01', end='2026-09-30', progress=False, auto_adjust=True)
d['Close'].to_csv('results/sp500_daily.csv')
print(d.shape)
