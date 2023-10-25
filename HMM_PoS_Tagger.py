import pickle
import numpy as np
import os
class HMM_PoS_Tagger:
    def __init__(self):
        self.trans_prob = {}
        self.emis_prob = {}

    def train(self, trainCorpus):
        print("Training...")

        # Data is a list of lists (sentences) of tuples word-tag

        for sentence in trainCorpus:

            prev_tag = "Start"  # Tag for calculating the initial probabilities
            for pair in sentence:
                word = pair[0]
                tag = pair[1]

                # Build the matrices by counting the number of appearances of each word and tag
                # Emission matrix
                if word not in self.emis_prob.keys():
                    self.emis_prob[word] = {}
                    self.emis_prob[word][tag] = 1
                else:
                    if tag not in self.emis_prob[word].keys():
                        self.emis_prob[word][tag] = 1
                    else:
                        self.emis_prob[word][tag] += 1

                # Transition matrix
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

        # Change from the count to the log probability of apparitions
        # Emission matrix
        for word in self.emis_prob.keys():
            total_apariciones = sum(self.emis_prob[word].values())
            for tag in self.emis_prob[word].keys():
                conteo_tag = self.emis_prob[word][tag]
                self.emis_prob[word][tag] = np.log(conteo_tag / total_apariciones)

        # Transition matrix
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
        # The prediction of the Hidden Markov Models depend on two assumptions
        # 1.- Markov assumption: the probability of a particular state only depends on the previous state.
        # 2.- Output independence: the probability of an output observation depends only on the current state.
        start = True
        tags = self.trans_prob.keys()
        best_path = []
        for word in sentence.split(" "):
            probs_act = {}
            if start:
                start = False
                previousTag = "Start"
                bestPreviousProb = 0

            for tag in tags:
                if tag!="Start":
                    emision = self.emis_prob[word][tag]
                    transition = self.trans_prob[previousTag][tag]
                    probs_act[tag] = bestPreviousProb + emision + transition

            bestPreviousProb = np.min(probs_act.values())
            previousTag = probs_act.keys()[np.argmin(probs_act.values())]
            best_path.append((word, previousTag))

        return best_path


    def probability(self, pair):
        print("Calculating probability...")

    def save_model(self, r):
        print("Saving model...")
        file = open(r, "wb")
        pickle.dump(self, file)
        file.close()