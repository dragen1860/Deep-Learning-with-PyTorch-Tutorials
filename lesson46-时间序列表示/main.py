import  torch
import  torchnlp

from torchnlp import word_to_vector






def main():

    # vec = word_to_vector.GloVe()
    vec = word_to_vector.BPEmb()



if __name__ == '__main__':
    main()