# coding = utf-8
import numpy as np
import pandas as pd

class Embedding:
    def __init__(self, matrix, vocabulary, word2index, normalize):
        '''
        Args:
            matrix          A numpy array, words associated with rows
            vocabulary      List of strings
            word2index      Dictionary mapping word to its index in 
                            "vocabulary".
            normalized      Boolean
        '''

        self.m = matrix
        self.normalized = normalize
        if normalize:
            self.normalize()
        self.dim = self.m.shape[1]
        self.wi = word2index
        self.iw = vocabulary
    

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]
        self.normalized = True

    @classmethod
    def from_fasttext_vec(cls,
                          path,
                          vocab_limit=None,
                          normalize=False):
        
        with open(path, 'r', encoding='utf-8') as f:
            
            vectors = [] # 2d-matrix, 一行就是一个word 的 embedding
            wi = {} # word 的 index
            iw = [] # 存放word

            first_line = f.readline().split() # vec 文件首行为 [vocab_size, dims]
            vocab_size = int(first_line[0])
            dim = int(first_line[1])

            if vocab_limit is None:
                vocab_limit = vocab_size
            
            for count in range(vocab_limit):
                line = f.readline().strip()

                parts = line.split() # 一行分割
                word = ' '.join(parts[:-dim]) # 取字
                vec = [float(x) for x in parts[-dim:]] # 取向量
                # 存储
                iw += [word]
                wi[word] = count
                vectors.append(vec)

        return cls(matrix=np.array(vectors),
                   vocabulary=iw,
                   word2index=wi,
                   normalize=normalize)            

def load_anew99(path='./anew99.csv'):
    anew = pd.read_csv(path, encoding='utf-8')
    anew.columns = ['Word', 'Valence', 'Arousal', 'Dominance']

    anew.set_index('Word', inplace=True)
    return anew


if __name__ == "__main__":
    emb = Embedding.from_fasttext_vec(path='./densifier_test.vec')

    print(emb.m)
    print(emb.wi)
    print(emb.iw)
    print(emb.dim)

    # anew test
    print(load_anew99())
