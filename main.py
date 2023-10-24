import HMM_PoS_Tagger
import pickle
from conllu import parse_incr

def conllu_preprocess(file):
    data_file = open(file, "r", encoding="utf-8")
    trainCorpus = []
    sentence = []
    for line in data_file:
        line = line.split('\t')
        if len(line) > 2:
            if '-' not in line[0]:
                if int(line[0]) == 1:
                    if sentence != []:
                        trainCorpus.append(sentence)
                    sentence = [(line[1], line[3])]
                else:
                    sentence.append((line[1], line[3]))
    return trainCorpus

def main():
    print('''WELCOME TO THE HMM Part of Speech Tagger

        Enter the number in order to:
            (1) Train polish model 
            (2) Train portuguese model
            (3) Exit

        By Ane García, Marcos Merino and Julia Wojciechowska\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Training polish model")
        trainCorpus = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-train.conllu")
        print(trainCorpus)
        main()

    elif int(eleccion) == 2:
        print("Training portuguese model")
        trainCorpus = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-train.conllu")
        print(trainCorpus)
        main()

    elif int(eleccion) == 3:
        print("EXIT...")
        return

    else:
        print("Incorrect selection\n\n")
        main()


if __name__ == "__main__":
    main()