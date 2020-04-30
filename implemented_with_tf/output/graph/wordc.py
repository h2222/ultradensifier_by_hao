# coding = utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False




def origin_vec(path):
    md = {}
    with open(path, 'r', encoding='utf_8_sig') as f:
        for i in f.readlines()[1:]:
            i = i.split(' ')
            s = i[0]
            x = i[1:]
            x = [float(i) for i in x]
            mean = sum(x)/len(x)
            md[s] = mean
    
    return md




# ultra dense
path = '../雷军评_lexicon.csv'
df = pd.read_csv(path, encoding='utf-8')
st = df['sentiment'][::30]

x = list(st)
y = [1 for i in range(len(st))]
n = list(df['word'].loc[st.index])


# origin space
# md = origin_vec('./雷军评.vec')
# print(md)
# x = list(md.values())[::30]
# y = [1 for i in range(len(x))]
# n = list(md.keys())[::30]


print(n)
print(x)


# x=[2.3,4.5,3,7,6.5,4,5.3]
# y=[1 for i in range(len())]
# n=np.arange(7)
 
fig,ax=plt.subplots()
ax.scatter(x,y,c='r')
 
for i,txt in enumerate(n):
     ax.annotate(txt,(x[i],y[i]))


plt.show()