import pandas as pd

df = pd.read_csv("")
df.dropna(how='all', inplace=True)