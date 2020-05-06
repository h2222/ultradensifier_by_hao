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




def get_result(path, s):
    sample = pd.read_csv(path, encoding='utf-8')
    print(sample)

    x = list(sample['sentiment'])
    y = [randint(-100, 100) for i in range(len(x))]
    n = list(sample['word'])

    return (x, y, n)

    # origin space
    # md = origin_vec('./雷军评.vec')
    # print(md)
    # x = list(md.values())[::50]
    # y = [1 for i in range(len(x))]
    # n = list(md.keys())[::50]

    # print(n)
    # print(x)


def graph2(item=None):
    plt.figure(figsize=(15, 15))
    plt.plot(item['5'][0], item['5'][1], 'bo', color='red', label='5 seed words', ms=0.001)
    plt.plot(item['10'][0], item['10'][1], 'bo', color='blue', label='10 seed words', ms=0.001)
    plt.plot(item['15'][0], item['15'][1], 'bo', color='green', label='15 seed words', ms=0.001)
    plt.title('The differences of domain specific lexicon with different seed words')
    plt.legend(loc='best')
    plt.xlabel('value of domain specific lexicon')
    clr = {'5':'red', '10':'blue', '15':'green'}

    for it in item:
        x = item[it][0]
        y = item[it][1]
        n = item[it][2]
        for i, txt in enumerate(n):
            plt.text(x=x[i], y=y[i], s=txt, fontsize=10, color=clr[it])

    plt.show()





if __name__ == "__main__":
    sw = ['5', '10', '15']

    item = {}
    for s in sw:
        # graph(r_path='../random_video/tot_lexicon'+s+'.csv', sw_path='../../Utils/source/cn_seed_v2_'+s+'.csv')
        result = get_result(path='../random_video/tot_lexicon'+s+'.csv', s=s)
        item[s] = result

    graph2(item=item)