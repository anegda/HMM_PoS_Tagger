# HMM_PoS_Tagger
This repository contains an implementation from scratch of the Hidden Markov Model for Part-of-speech Tagging.

## :hammer: Project Structure
    Corpus
    ├── Polish                                  # Corpus used for Polish PoS task
    │   ├── pl_lfg-ud-dev.conllu
    │   ├── pl_lfg-ud-test.conllu
    │   └── pl_lfg-ud-train.conllu
    ├── Polish                                  # Corpus used for Portuguese PoS task
    │   ├── pt_petrogold-ud-dev.conllu
    │   ├── pt_petrogold-ud-test.conllu
    └── └── pt_petrogold-ud-train.conllu

    Jupyter Notebooks                           # Jupyter notebooks where the experimentation is done
    ├── Experimentation_pl.ipynb
    └── Experimentation_pt.ipynb

    Models                                      # Models created
    ├── HMM_PoS_Tagger_pl
    └── HMM_PoS_Tagger_pt

    HMM_PoS_Tagger.py                           # Python class with the HMM model (methods, parameters...)
    main.py                                     # Main python script with all the posible options (training, evaluation, prediction...)

## :speech_balloon: Explanation of the Corpus
Both Corpus are obtain from the official site of Universal Dependencies (UD) (https://universaldependencies.org/). UD is a framework for consistent annotation of grammar across 100+ different human languages. 

This annotation includes the parts of speech tags that refer to the syntactic role of each word in a sentence. The UD includes a total of 17 tags where we can find:
* Adjectives (ADJ): Noun modifiers describing properties.
* Adverb (ADV): Verb modifiers of time, place, manner.
* Noun (NOUN): words for persons, places, things, etc.
* Proper noun (PROPN): name of person, organization, place, etc.
* Interjection (INTJ): exclamation, greeting, yes/no response, etc.
* Adposition (ADP): marks a noun's spacial, temporal or other relation.
* Auxiliary (AUX): helping verb marking tense, aspect, mood, etc.
* Coordinating Conjuction (CCONJ): joins teo phrases/clauses.
* Determiner (DET): marks noun phrase properties.
* Numeral (NUM)
* Particle (PART): a preposition-like form used together with a verb.
* Pronoun (PRON): a shorthand for referring to an entity or event.
* Subordinating Conjunction (SCONJ): joins a main clause with subordinate clause.
* Punctuation (PUNCT)
* Symbols (SYM)
* Other X

In this case we have chosen *Polish* and *Portuguese* as the languages to study. The Corpus selected in the case of Polish is the *LFG Enhanced UD treebank* that consists of 17,246 sentences split into 13,744 trees in training, 1745 trees for development and 1727 trees for testing. Some examples of the tagging:
* ADJ: rządowe, jeden, nowy...
* ADV: potem, jak, teraz...
* NOUN: lud, bezpłodnością, pogróżki...
* PROPN: Adam, Aga, Angus...
* INTJ: ha, y, o...
* ADP: o, po, na...
* AUX: będzie, by, m...
* CCONJ: i, ale, a...
* DET: tej, wielu, swoim...
* NUM: siedem, 1, dwie...
* PART: podobno, około, też...
* PRON: mnie, się, wam...
* SCONJ: że, choć, jeżeli...

UD_Portuguese-PetroGold is a fully revised treebank which consists of academic texts from the oil & gas domain in Brazilian Portuguese. The Corpus consists of 7170 train sentences, 1039 test sentences and 737 dev sentences. Some examples of the tagging:
* ADJ: novos, azul, privadas...
* ADV: consideravelmente, quantitativamente, mais...
* NOUN: estudos, petróleo, decomposiçao...
* PROPN: VAZ, Carbono, Oxigênio... 
* ADP: de, em, para...
* AUX: são, esta, sendo...
* CCONJ: ou, e, como...
* DET: as, um, aquele...
* NUM: 2002, dois, 4...
* PRON: isso, se, isto...
* SCONJ: a, que, pois...

## :desktop_computer: Guide for use
The project contains a main.py file that when executed you can access to the principal options of the model: training, prediction of tags for a sentence, evaluation, etc.

* Option 1: training the model. You can decide between the Polish or Portuguese corpus, select the threshold of unknown tokens and enable the use of suffix matrix. After the training the model is automatically saved.
* Option 2: the user writes a sentence in Polish and the pretrained model tags each word.
* Option 3: the user writes a sentence in Portuguese and the pretrained model tags each word.
* Option 4: complete evaluation of the chosen model.
* Option 5: sweep of the unknown token threshold hyperparameter for Polish corpus.
* Option 6: sweep of the unknown token threshold hyperparameter for Portuguese corpus.
* Option 7: exit the program 