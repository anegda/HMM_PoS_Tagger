import HMM_PoS_Tagger
import pickle
import re
import os


def conllu_preprocess(file):

    trainCorpus = []
    sentence = []
    multi_token_dict = {}

    with open(file, 'r', encoding="utf-8") as data_file:
        lines = data_file.readlines()
        total_lines = len(lines)

        for i, line in enumerate(lines):
            if i < total_lines - 2:
                line = line.split('\t')
                if len(line) > 2:
                    if '-' not in line[0]:
                        if int(line[0]) == 1:
                            if sentence != []:
                                trainCorpus.append(sentence)
                            sentence = [(line[1].lower(), line[3])]
                        else:
                            sentence.append((line[1].lower(), line[3]))
                    else:
                        nextLine1 = lines[i + 1]
                        nextLine2 = lines[i + 2]
                        nextLine1 = nextLine1.split('\t')
                        nextLine2 = nextLine2.split('\t')
                        multi_token_dict[line[1].lower()] = nextLine1[1].lower()+ " " + nextLine2[1].lower()

    return trainCorpus, multi_token_dict

def unk_sweep(train_ruta, test_ruta, unk_array):

  for unk in unk_array:
    # First we obtain the train and test corpus, and the multiword dictionaries for each
    trainCorpus, trainCorpus_multi_tokens = conllu_preprocess(train_ruta)
    testCorpus, testCorpus_multi_tokens = conllu_preprocess(test_ruta)

    # Generate and train the model
    tagger = HMM_PoS_Tagger.HMM_PoS_Tagger()
    tagger.setMultiTokensDict(trainCorpus_multi_tokens)
    tagger.train(trainCorpus, unk_value=unk)

    # Evaluate the model
    print("#### UNK-VALUE:", unk, "####")
    tagger.evaluate(testCorpus)
    print("\n")

def main():
    print('''WELCOME TO THE HMM Part of Speech Tagger

        Enter the number in order to:
            (1) Train a model 
            (2) Predict Tags for a Sentence in Polish
            (3) Predict Tags for a Sentence in Portuguese
            (4) Evaluate a model
            
            ----- FUNCTIONS FOR DEVELOPMENT -----
            (5) UNK SWEEP for Polish
            (6) UNK SWEEP for Portuguese       
            -------------------------------------
            
            (7) Exit

        By Ane García, Marcos Merino and Julia Wojciechowska\n''')

    eleccion = input()

    if not eleccion.isdigit() or int(eleccion) > 7:
        print("Incorrect selection\n\n")
        main()

    elif int(eleccion) == 1:
        print(''' 
        Select which model you would like to train:
            (1) Polish
            (2) Portuguese
            (Other) Back
        ''')

        eleccion = input()

        if int(eleccion) == 1:

            print("Input number of UNK tokens (default: 3)")
            unks = input()
            if not unks.isdigit():
                unks = 3

            print("Value selected:", unks, "\n")

            print("Training Polish model")
            trainCorpus, trainCorpus_multi_tokens = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-train.conllu")
            tagger = HMM_PoS_Tagger.HMM_PoS_Tagger()
            tagger.train(trainCorpus, unk_value=int(unks))
            if not os.path.exists("./Models"):
                os.mkdir("./Models")
            tagger.save_model("./Models/pl_HMM_PoS_tagger.sav")
            print("Model trained\n")
            main()

        elif int(eleccion) == 2:

            print("Input number of UNK tokens (default: 3)")
            unks = input()
            if not unks.isdigit():
                unks = 3

            print("Value selected:", unks, "\n")
            print("Training Portuguese model")
            trainCorpus, trainCorpus_multi_tokens = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-train.conllu")
            tagger = HMM_PoS_Tagger.HMM_PoS_Tagger()
            tagger.setMultiTokensDict(trainCorpus_multi_tokens)
            tagger.train(trainCorpus, unk_value=int(unks))
            if not os.path.exists("./Models"):
                os.mkdir("./Models")
            tagger.save_model("./Models/pt_HMM_PoS_tagger.sav")
            print("Model trained\n")
            main()

        else:
            print("Going back...\n\n")
            main()


    elif int(eleccion) == 2:
        sentence = input("Input a sentence in Polish: ").lower()
        file = open("./Models/pl_HMM_PoS_tagger.sav", "rb")
        tagger = pickle.load(file)
        file.close()

        words = re.findall(r'\b\w+\b|[.,!?;:(){}¿¡|]', sentence)
        sentence = " ".join(words)
        print("\nThe predicted sequence of tags is:")
        print(str(tagger.predict(sentence)) + "\n\n")
        main()


    elif int(eleccion) == 3:
        #a caracterização estrutural para a porção
        sentence = input("Introduce a sentence in Portuguese: ").lower()

        file = open("./Models/pt_HMM_PoS_tagger.sav", "rb")
        tagger = pickle.load(file)

        words = re.findall(r'\b\w+\b|[.,!?;:{}"()¿¡|]', sentence)

        for i in range(len(words)):
            word = words[i]
            if word in tagger.multi_word_tokens:
                words[i] = tagger.multi_word_tokens[word]

        sentence = " ".join(words)

        file.close()
        print("\nThe predicted sequence of tags is:")
        print(str(tagger.predict(sentence)) + "\n")
        main()


    elif int(eleccion) == 4:

        print(''' 
            Select which model you would like to evaluate:
                (1) Polish
                (2) Portuguese
                (Other) Back
            ''')

        eleccion = input()

        if int(eleccion) == 1:
            if os.path.exists("./Models/pl_HMM_PoS_tagger.sav"):

                print(''' 
                            Select which an option:
                                (1) Check metrics' scores.
                                (2) Check the top 10 sentences with the highest % of incorrect predictions.
                                (Other) Back
                            ''')

                eleccion = input()

                if int(eleccion) == 1:

                    file = open("./Models/pl_HMM_PoS_tagger.sav", "rb")
                    tagger = pickle.load(file)
                    file.close()

                    testCorpus, testCorpus_multi_tokens = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-dev.conllu")
                    tagger.evaluate(testCorpus)

                else:
                    file = open("./Models/pl_HMM_PoS_tagger.sav", "rb")
                    tagger = pickle.load(file)
                    file.close()
                    testCorpus, testCorpus_multi_tokens = conllu_preprocess("./Corpus/Polish/pl_lfg-ud-test.conllu")
                    tagger.top10worstPredictions(testCorpus)

                main()
            else:
                print("No trained model for Polish exists yet. Please train a model first\n\n")

        if int(eleccion) == 2:
            if os.path.exists("./Models/pt_HMM_PoS_tagger.sav"):

                print(''' 
                                            Select which an option:
                                                (1) Check metrics' scores.
                                                (2) Check the top 10 sentences with the highest % of incorrect predictions.
                                                (Other) Back
                                            ''')

                eleccion = input()

                if int(eleccion) == 1:
                    file = open("./Models/pt_HMM_PoS_tagger.sav", "rb")
                    tagger = pickle.load(file)
                    file.close()

                    testCorpus, testCorpus_multi_tokens = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-dev.conllu")
                    tagger.evaluate(testCorpus)

                else:
                    file = open("./Models/pt_HMM_PoS_tagger.sav", "rb")
                    tagger = pickle.load(file)
                    file.close()
                    testCorpus, testCorpus_multi_tokens = conllu_preprocess("./Corpus/Portuguese/pt_petrogold-ud-test.conllu")
                    tagger.top10worstPredictions(testCorpus)

                main()
            else:
                print("No trained model for Portuguese exists yet. Please train a model first\n\n")

        else:
            print("Going back...\n\n")
            main()

        main()

    elif int(eleccion) == 5:
        print("UNK SWEEP - Input however many values for the UNK sweep, separated by spaces")
        unk_array = input().strip().split(" ")
        unk_array = [int(x) for x in unk_array]
        unk_sweep("./Corpus/Polish/pl_lfg-ud-train.conllu", "./Corpus/Polish/pl_lfg-ud-dev.conllu", unk_array)

        main()


    elif int(eleccion) == 6:
        print("UNK SWEEP - Input however many values for the UNK sweep, separated by spaces")
        unk_array = input().strip().split(" ")
        unk_array = [int(x) for x in unk_array]
        unk_sweep("./Corpus/Portuguese/pt_petrogold-ud-train.conllu", "./Corpus/Portuguese/pt_petrogold-ud-dev.conllu", unk_array)

        main()


    elif int(eleccion) == 7:
        print("Exiting...")
        return


if __name__ == "__main__":
    main()
