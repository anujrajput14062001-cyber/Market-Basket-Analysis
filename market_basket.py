import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("transactions.csv")
data = data.fillna('')

te = TransactionEncoder()
te_data = te.fit(data.values).transform(data.values)
df = pd.DataFrame(te_data, columns=te.columns_)

frequent_items = apriori(df, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

print(rules[['antecedents','consequents','confidence']])
