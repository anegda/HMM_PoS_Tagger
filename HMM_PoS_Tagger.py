import pickle

class HMM_PoS_Tagger:
    def __init__(self):
        self.trans_prob = []
        self.emis_prob = []

    def train(self, trainCorpus):
        print("Training...")

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