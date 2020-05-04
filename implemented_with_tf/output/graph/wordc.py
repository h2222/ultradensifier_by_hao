# coding = utf-8

from random import randint
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

def graph(r_path, sw_path):
    df = pd.read_csv(r_path, encoding='utf-8')
    df = df.set_index('word')
    sw = pd.read_csv(sw_path, encoding='utf-8')
    sw = sw.set_index('word')
    sample = df.loc[sw.index]
    print(sample)

    x = list(sample['sentiment'])
    y = [1 for i in range(len(x))]
    n = list(sample.index)

    # origin space
    # md = origin_vec('./雷军评.vec')
    # print(md)
    # x = list(md.values())[::50]
    # y = [1 for i in range(len(x))]
    # n = list(md.keys())[::50]

    # print(n)
    # print(x)
    
    # fig,ax=plt.subplots()
    # ax.scatter(x,y,c='r')

    plt.plot(x, y, 'bo', ms=0.001)
    plt.title('distribution of seed_word based on domain specific lexicon embedding')
    plt.xlabel('value of domain specific lexicon')

    for i, txt in enumerate(n):
        ran = randint(-15, 15)
        plt.text(x=x[i], y=y[i]+0.001*ran, s=txt, fontsize=15, color='mediumvioletred')
        # ax.annotate(txt,(x[i],y[i]))
    plt.show()



if __name__ == "__main__":
    sw = ['5', '10',  '15']

    for s in sw:
        graph(r_path='../random_video/tot_lexicon'+s+'.csv', 
              sw_path='../../Utils/source/cn_seed_v2_'+s+'.csv')