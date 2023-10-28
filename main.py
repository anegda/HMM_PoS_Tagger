import HMM_PoS_Tagger
import pickle


def conllu_preprocess(file):

    data_file = open(file, "r", encoding="utf-8")
    trainCorpus = []
    corpus = []
    sentence = []
    for line in data_file:
        line = line.split('\t')
        if len(line) > 2:
            if '-' not in line[0]:
                if int(line[0]) == 1:
                    if sentence != []:
                        trainCorpus.append(sentence)
                    sentence = [(line[1].lower(), line[3])]
                else:
                    sentence.append((line[1].lower(), line[3]))
    corpus.append(sentence)
    return trainCorpus


def main():
    print('''WELCOME TO THE HMM Part of Speech Tagger

        Enter the number in order to:
            (1) Train polish model 
            (2) Train portuguese model
            (3) Predict Sentence in Polish
            (4) Predict Sentence in Portuguese
            (5) Evaluate Portuguese test
            (6) Evaluate Polish test
            (7) Exit

        By Ane García, Marcos Merino and Julia Wojciechowska\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Training polish model")
        trainCorpus = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-train.conllu")
        testCorpus = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-test.conllu")
        tagger = HMM_PoS_Tagger.HMM_PoS_Tagger()
        tagger.train(trainCorpus)
        tagger.save_model("./Models/pl_HMM_PoS_tagger.sav")
        tagger.evaluate(trainCorpus)
        main()

    elif int(eleccion) == 2:
        print("Training portuguese model")
        trainCorpus = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-train.conllu")
        testCorpus = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-test.conllu")
        tagger = HMM_PoS_Tagger.HMM_PoS_Tagger()
        tagger.train(trainCorpus)
        tagger.save_model("./Models/pt_HMM_PoS_tagger.sav")
        tagger.evaluate(trainCorpus)
        main()

    elif int(eleccion) == 3:
        sentence = input("Introduce a sentence in Polish: ").lower()
        file = open("./Models/pl_HMM_PoS_tagger.sav", "rb")
        tagger = pickle.load(file)
        file.close()
        print(str(tagger.predict(sentence)) + "\n")
        main()

    elif int(eleccion) == 4:
        #a caracterização estrutural para a porção
        sentence = input("Introduce a sentence in Portuguese: ").lower()
        file = open("./Models/pt_HMM_PoS_tagger.sav", "rb")
        tagger = pickle.load(file)
        file.close()
        print(str(tagger.predict(sentence)) + "\n")
        main()

    elif int(eleccion) == 5:
        testCorpus = [
            [("a", "DET"), ("bacia", "PROPN"), ("de", "ADP"), ("pelotas", "PROPN"), ("é", "AUX"), ("a", "DET"),
             ("mas", "ADV"), ("meridional", "ADJ")],
            [("a", "DET"), ("caracterização", "NOUN"), ("estrutural", "ADJ"), ("para", "ADP"), ("a", "DET"),
             ("porção", "NOUN")]
        ]
        file = open("./Models/pt_HMM_PoS_tagger.sav", "rb")
        tagger = pickle.load(file)
        file.close()
        tagger.recall(testCorpus)

    elif int(eleccion) == 6:
        print("Unimplemented...")
        return

    elif int(eleccion) == 7:
        print("Exiting...")
        return

    else:
        print("Incorrect selection\n\n")
        main()


if __name__ == "__main__":
    main()
