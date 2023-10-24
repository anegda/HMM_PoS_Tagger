import pickle
import numpy as np
class HMM_PoS_Tagger:
    def __init__(self):
        self.trans_prob = {}
        self.emis_prob = {}

    def train(self, trainCorpus):
        print("Training...")

        # Los datos de entrada son una lista de listas de tuplas word-tag

        for sentence in trainCorpus:

            prev_tag = "Start"  # Para calcular las probabilidades de inicio
            for pair in sentence:
                word = pair[0]
                tag = pair[1]

                # Formar las matrices contando el numero de apariciones de words y de tags
                if word not in self.emis_prob.keys():
                    self.emis_prob[word] = {}
                    self.emis_prob[word][tag] = 1
                else:
                    if tag not in self.emis_prob[word].keys():
                        self.emis_prob[word][tag] = 1
                    else:
                        self.emis_prob[word][tag] += 1

                if prev_tag not in self.trans_prob.keys():
                    self.trans_prob[prev_tag] = {}
                    self.trans_prob[prev_tag][tag] = 1
                else:
                    if tag not in self.trans_prob[prev_tag].keys():
                        self.trans_prob[prev_tag][tag] = 1
                    else:
                        self.trans_prob[prev_tag][tag] += 1
                prev_tag = tag

            # Al acabar la frase añadir la probabilidad de tag terminal
            if (prev_tag, "Stop") not in self.trans_prob.keys():
                self.trans_prob[prev_tag] = {}
                self.trans_prob[prev_tag]["Stop"] = 1
            else:
                self.trans_prob[prev_tag]["Stop"] += 1

        # Transformar los valores count en probabilidades de aparición
        for word in self.emis_prob.keys():
            total_apariciones = sum(self.emis_prob[word].values())
            for tag in self.emis_prob[word].keys():
                conteo_tag = self.emis_prob[word][tag]
                self.emis_prob[word][tag] = np.log(conteo_tag / total_apariciones)

        for prev_tag in self.trans_prob.keys():
            total_apariciones = sum(self.trans_prob[prev_tag].values())
            for tag in self.trans_prob[prev_tag].keys():
                conteo_tag = self.trans_prob[prev_tag][tag]
                self.trans_prob[prev_tag][tag] = np.log(conteo_tag / total_apariciones)

    def evaluate(self, testCorpus):
        print("Evaluating...")

    def accuracy(self, testCorpus):
        return 0

    def precision(self, testCorpus):
        return 0

    def recall(self, testCorpus):
        return 0

    def confusion_matrix(self, testCorpus):
        print('Confusion matrix')

    def predict(self, sentence):
        print("Predicting...")

    def probability(self, pair):
        print("Calculating probability...")

    def save_model(self, r):
        print("Saving model...")
        file = open(r, "wb")
        pickle.dump(self, file)
        file.close()