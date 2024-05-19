from src.eval import performance
import pandas as pd

df = pd.read_pickle(r"/gpfs/bwfor/home/ma/ma_ma/ma_kyupark/is809/data/rawdata/liwcvalid.pkl")
perf = performance(df)
print(perf)
perf.to_pickle(r"/gpfs/bwfor/home/ma/ma_ma/ma_kyupark/is809/data/output/validation/liwcperf.pkl")