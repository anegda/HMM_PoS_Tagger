import pickle
import numpy as np
import os
import main
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate

class HMM_PoS_Tagger:
    def __init__(self):
        ud_pos_tags = ["Start", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
        ud_prev_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "Stop"]
        # Initialization of every transition log-probability to negative infinity
        # 'Start' tag only present as a previous tag, 'Stop' as a final tag
        self.trans_prob = {tag: {subtag: 0.0001 for subtag in ud_prev_tags} for tag in ud_pos_tags}
        self.emis_prob = {}
        self.multi_word_tokens = {}

    def setMultiTokensDict(self,multi_word_tokens):
        self.multi_word_tokens = multi_word_tokens
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

                # Transition matrix, check if cell value is negative infinity first
                self.trans_prob[prev_tag][tag] += 1
                prev_tag = tag

            # Add the probability of the tag being terminal to the transition matrix
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

        # There are combinations of word-tag that might not appear in the training corpus
        # To deal with those cases, we must check if the entry exists in the matrices and if not, assign a
        # negative infinite log-probability

    def evaluate(self, testCorpus):
        print("Evaluating...\n")

        # First, obtain the gold metrics to compare the predictions
        gold_tags = []
        test_sentences = []

        for sent in testCorpus:
            words = []
            for pair in sent:
                gold_tags.append(pair[1])
                words.append(pair[0])
            test_sentences.append(" ".join(words))

        # Sentences are in a list of lists
        # Gold tags are just a list

        # Then, we obtain de predictions of the model using the predict() method
        predictions = []
        for sentence in test_sentences:
            best_path = self.predict(sentence)
            # The predict method returns a list of tuples word-tag, we just need the tags
            predictions.extend([tag for word, tag in best_path])

        gold_counts = Counter(gold_tags)
        predictions_counts = Counter(predictions)
        tags = set(gold_tags).union(set(predictions))

        countsTable = []

        for tag in tags:
            gold_counts_tag = gold_counts.get(tag, 0)
            predictions_counts_tag = predictions_counts.get(tag, 0)
            diff = gold_counts_tag - predictions_counts_tag
            if diff < 0:
                diff = abs(diff)
            countsTable.append([tag, gold_counts_tag, predictions_counts_tag, diff])

        metricsResuts = [["Accuracy",str(self.accuracy(gold_tags, predictions))] , ["Precison",str(self.precision(gold_tags, predictions))], ["Recall",str(self.recall(gold_tags, predictions))],["F1 Score",str(self.f1_score(gold_tags, predictions))]]
        tableCounts = tabulate(countsTable, ["Tag", "Gold counts", "Prediction counts", "Difference"], tablefmt="grid")
        tableMetrics = tabulate(metricsResuts, ["Metric", "Score"], tablefmt="grid")

        print("The resoults of the evaluations are:\n")
        print("Metrics:\n")
        print(tableMetrics)
        print("\nCounts' differences:\n")
        print(tableCounts + "\n")




        self.confusion_matrix(gold_tags, predictions)

    def accuracy(self, gold_tags, predictions):


        # At this point, both gold and pred are lists of tags
        score = len([1 for x, y in zip(gold_tags, predictions) if x == y])
        total = len(gold_tags)
        acc = accuracy_score(gold_tags, predictions)



        #print("GOLD:\t\t", gold_tags)
        #print("PREDICTIONS:", predictions)
        #print("------------------------------------------------")
        print("SCORE:", score, "/", total)
        #print("ACCURACY:", acc)
        return acc

    def precision(self, gold_tags, predictions):

        # At this point, both gold and pred are lists of tags
        precision = precision_score(gold_tags, predictions, average="weighted")

        #print("GOLD:\t\t", gold_tags)
        #print("PREDICTIONS:", predictions)
        #print("PRECISION", precision)
        return precision

    def recall(self, gold_tags, predictions):

        # At this point, both gold and pred are lists of tags
        recall = recall_score(gold_tags, predictions, average="weighted")

        #print("GOLD:\t\t", gold_tags)
        #print("PREDICTIONS:", predictions)
        #print("RECALL:", recall)
        return recall

    def f1_score(self, gold_tags, predictions):

        # At this point, both gold and pred are lists of tags
        fscore = f1_score(gold_tags, predictions, average="weighted")

        #print("GOLD:\t\t", gold_tags)
        #print("PREDICTIONS:", predictions)
        #print("F1_SCORE:", fscore)
        return fscore

    def confusion_matrix(self, gold_tags, predictions):
        #print('Confusion matrix')

        # At this point, both gold and pred are lists of tags
        conf_matrix = confusion_matrix(gold_tags, predictions)

        #print("GOLD:\t\t", gold_tags)
        #print("PREDICTIONS:", predictions)
        #print("CONFUSION MATRIX:")
        #print(conf_matrix)

        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=sorted(set(gold_tags).union(set(predictions))))
        fig, ax = plt.subplots()
        display.plot(ax=ax)
        ax.set_xticklabels(display.display_labels, rotation=-45)
        ax.set_title("Confusion matrix", loc="center")
        plt.show()

        return conf_matrix

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
                    # EMISSION PROBABILITIES
                    if word in self.emis_prob.keys():
                        if tag in self.emis_prob[word].keys():
                            emission = self.emis_prob[word][tag]
                        else:
                            emission = float('-inf')
                    else:
                        if tag in self.emis_prob['UNK'].keys():
                            emission = self.emis_prob['UNK'][tag]
                        else:
                            emission = float('-inf')

                    # TRANSITION PROBABILITIES
                    transition = self.trans_prob[previousTag][tag]
                    probs_act[tag] = bestPreviousProb + emission + transition

            bestPreviousProb = np.max(list(probs_act.values()))
            previousTag = list(probs_act.keys())[np.argmax(list(probs_act.values()))]
            best_path.append((word, previousTag))

        return best_path

    def probability(self, pair):
        print("Calculating probability...")

    def save_model(self, r):
        print("Saving model...")
        file = open(r, "wb")
        pickle.dump(self, file)
        file.close()