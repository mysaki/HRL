import pandas as pd
import numpy as np

f = pd.DataFrame(data=np.arange(10).reshape((5, 2)), columns=(["city", "name"]),index=(["a","b","c","d","e"]))
print(f.loc[["b"],["city"]])
print(f.reindex(columns=["city"]))
d = f.reindex(columns=["city"])
print(d)
print(f)
print(f["city"]>2)
print(f.loc["b":"d","name"])