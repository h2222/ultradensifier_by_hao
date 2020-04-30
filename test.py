

import pandas as pd


df2 = pd.DataFrame(data={'sentiment':['+', '+', '-', '-']}, index = ['a', 'b', 'c', 'd'])

df = pd.DataFrame(data={'sentiment':[1, 2, 3]}, index=['a', 'b', 'c'])

print(df2)

word = ['b', 'c', 'd']
 

for w in word:
    if not w in df.index:
        df.loc[w] = 'invalid'


def rule(x):
    if x =='invalid':
        return x
    else:
        if x > 1.5:
            return '+'
        else:
            return '-'


df['sentiment'] = df['sentiment'].apply(rule)

dp = df.loc[df['sentiment'] == 'invalid'].index


df2 = df2.drop(dp)


print(iter(list(df2.index)))




