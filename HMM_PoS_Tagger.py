import pickle
import numpy as np
from sklearn.metrics import classification_report
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
        ud_pos_tags = ["Start", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
                       "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
        ud_prev_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                        "PUNCT", "SCONJ", "SYM", "VERB", "X", "Stop"]
        # Initialization of every transition log-probability to negative infinity
        # 'Start' tag only present as a previous tag, 'Stop' as a final tag
        self.trans_prob = {tag: {subtag: 0.0001 for subtag in ud_prev_tags} for tag in ud_pos_tags}
        self.emis_prob = {}
        self.suffix_prob = {}
        self.multi_word_tokens = {}

    def setMultiTokensDict(self, multi_word_tokens):
        self.multi_word_tokens = multi_word_tokens

    def infrequent_words_to_unk(self, trainCorpus, unk_value):
        words_count = {}

        # we calculate the distribution for each word
        for sentence in trainCorpus:
            for pair in sentence:
                if pair[0] in words_count:
                    words_count[pair[0]] += 1
                else:
                    words_count[pair[0]] = 1

        # we replace all infrequent words with a single 'UNK' token
        new_trainCorpus = []
        for sentence in trainCorpus:
            new_sentence = []
            for pair in sentence:
                if words_count[pair[0]] < unk_value:
                    new_sentence.append(('UNK', pair[1]))
                else:
                    new_sentence.append(pair)
            new_trainCorpus.append(new_sentence)

        return new_trainCorpus

    def suffix_matrix(self, trainCorpus):
        for sentence in trainCorpus:
            for pair in sentence:
                if (len(pair[0]) > 4):
                    suffix = pair[0][len(pair[0]) - 4:]
                    tag = pair[1]

                    # Build the matrix by counting the number of appearances of each suffix and tag
                    # Suffix matrix
                    if suffix not in self.suffix_prob.keys():
                        self.suffix_prob[suffix] = {}
                        self.suffix_prob[suffix][tag] = 1
                    else:
                        if tag not in self.suffix_prob[suffix].keys():
                            self.suffix_prob[suffix][tag] = 1
                        else:
                            self.suffix_prob[suffix][tag] += 1

        # Change from the count to the log probability of apparitions
        # Suffix matrix
        for suf in self.suffix_prob.keys():
            total_apariciones = sum(self.suffix_prob[suf].values())
            for tag in self.suffix_prob[suf].keys():
                conteo_tag = self.suffix_prob[suf][tag]
                self.suffix_prob[suf][tag] = np.log(conteo_tag / total_apariciones)

    def train(self, trainCorpus, suffix=False, unk_value=3):
        print("Training...")

        # Data is a list of lists (sentences) of tuples word-tag
        if suffix:
            self.suffix_matrix(trainCorpus)

        trainCorpus = self.infrequent_words_to_unk(trainCorpus, unk_value=unk_value)

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

    def eval(self, testCorpus):
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

        return gold_tags, predictions

    def top10worstPredictions(self, testCorpus):

        i = 0
        countsErrors = []
        countSErrorsUnit = []

        gold_tags, predictions = self.eval(testCorpus)

        for sent in testCorpus:
            countErrors = 0
            errors = []
            countSErrorsUnit.append(sent)
            for pair in sent:
                if pair[1] != predictions[i]:
                    errors.append((pair[0], predictions[i]))
                    countErrors += 1
                i += 1
            countSErrorsUnit.append(errors)
            countSErrorsUnit.append(countErrors / len(sent))
            countsErrors.append(countSErrorsUnit)
            countSErrorsUnit = []

        top10positions = [index for index, _ in sorted(enumerate(countsErrors), key=lambda x: x[1][2], reverse=True)[:10]]
        top10sentences = [countsErrors[pos] for pos in top10positions]

        tableCounts = tabulate(top10sentences, ["Real sentence and tags", "Incorrect tags", "Error rate"], tablefmt="grid")
        print(tableCounts)

    def evaluate(self, testCorpus):
        print("Evaluating...\n")

        gold_tags, predictions = self.eval(testCorpus)

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

        metricsResuts = [["Accuracy", str(self.accuracy(gold_tags, predictions))],
                         ["Precison", str(self.precision(gold_tags, predictions))],
                         ["Recall", str(self.recall(gold_tags, predictions))],
                         ["F1 Score", str(self.f1_score(gold_tags, predictions))]]
        tableCounts = tabulate(countsTable, ["Tag", "Gold counts", "Prediction counts", "Difference"], tablefmt="grid")
        tableMetrics = tabulate(metricsResuts, ["Metric", "Score"], tablefmt="grid")

        print("The results of the evaluations are:\n")
        print("Metrics:\n")
        print(tableMetrics)
        print("\nCounts' differences:\n")
        print(tableCounts + "\n")

        self.confusion_matrix(gold_tags, predictions)

    def evaluate_summary(self, testCorpus):
        # First, obtain the gold metrics to compare the predictions
        gold_tags, predictions = self.eval(testCorpus)

        metricsResults = [["Accuracy", str(self.accuracy(gold_tags, predictions))],
                          ["Precison", str(self.precision(gold_tags, predictions))],
                          ["Recall", str(self.recall(gold_tags, predictions))],
                          ["F1 Score", str(self.f1_score(gold_tags, predictions))]]
        tableMetrics = tabulate(metricsResults, ["Metric", "Score"], tablefmt="grid")

        print("Metrics:\n")
        print(tableMetrics)

    def evaluate_per_tag(self, testCorpus):
        # First, obtain the gold metrics to compare the predictions
        gold_tags, predictions = self.eval(testCorpus)

        print(classification_report(gold_tags, predictions, digits=4))

    def accuracy(self, gold_tags, predictions):
        # At this point, both gold and pred are lists of tags
        score = len([1 for x, y in zip(gold_tags, predictions) if x == y])
        total = len(gold_tags)
        acc = accuracy_score(gold_tags, predictions)

        print("SCORE:", score, "/", total)
        return acc

    def precision(self, gold_tags, predictions):
        # At this point, both gold and pred are lists of tags
        precision = precision_score(gold_tags, predictions, average="weighted")

        return precision

    def recall(self, gold_tags, predictions):
        # At this point, both gold and pred are lists of tags
        recall = recall_score(gold_tags, predictions, average="weighted")

        return recall

    def f1_score(self, gold_tags, predictions):
        # At this point, both gold and pred are lists of tags
        fscore = f1_score(gold_tags, predictions, average="weighted")

        return fscore

    def confusion_matrix(self, gold_tags, predictions):
        # At this point, both gold and pred are lists of tags
        conf_matrix = confusion_matrix(gold_tags, predictions)

        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                         display_labels=sorted(set(gold_tags).union(set(predictions))))
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
                if tag != "Start":
                    # EMISSION PROBABILITIES
                    if word in self.emis_prob.keys():
                        if tag in self.emis_prob[word].keys():
                            emission = self.emis_prob[word][tag]
                        else:
                            emission = float('-inf')
                    else:
                        if (self.suffix_prob != {} and len(word) > 4) and word[
                                                                          len(word) - 4:] in self.suffix_prob.keys() and tag in \
                                self.suffix_prob[word[len(word) - 4:]].keys():
                            emission = self.suffix_prob[word[len(word) - 4:]][tag]
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