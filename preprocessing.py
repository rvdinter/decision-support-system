import os
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Preprocessor:
    def __init__(self):  # , buffer_size, batch_size, pad_title, pad_abs
        nltk.download('stopwords')
        self._vocab_size = 0
        self._max_length = 600
        self.vocab = Counter()
        self.WV_DIM = 100
        self.vocab_tokens = None
        self._embedding_model = None
        self._word_index = {}

    def get_tf_datasets(self, train_path, test_path, embedding_path, buffer_size, batch_size, oversampling=True,
                        show_imbalance=False):
        print('Loading train and test set')
        self._load_train_test(train_path, test_path, batch_size)
        
        if show_imbalance:
            self._show_class_imbalance(self._train)

        print('Loading and creating embeddings, this may take a while')
        self._create_embedding_matrix(embedding_path)
        
        print('Creating datasets')
        self._create_tf_datasets(buffer_size, batch_size, oversampling)
        return self._train_ds, self._test_ds, self._steps_per_epoch

    def get_max_sequence_length(self):
        return self._max_length

    def calculate_steps_per_epoch(self, df, batch_size):
        """
        Calculate the steps per epoch we must perform to see every positive at least once as we are oversampling
        """
        neg, _ = np.bincount(df['label'])
        return np.ceil(2.0 * neg / batch_size)

    def _show_class_imbalance(self, df):
        """
        show if there is a class imbalance
        """
        neg, pos = np.bincount(df['label'])
        total = neg + pos
        print('Samples in training set:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))

        plt.figure(figsize=(10, 8))
        sns.countplot(df['label'])
        plt.title('Training set distribution before oversampling')
        plt.show()

    def _load_train_test(self, train_path, test_path, batch_size):
        self._train = self._load_dataset(train_path)
        self._test = self._load_dataset(test_path)

        # Create a vocabulary consisting of all data samples
        full_dataset = pd.concat([self._train, self._test])
        self._create_vocab(full_dataset)
        self._steps_per_epoch = self.calculate_steps_per_epoch(full_dataset, batch_size)

        # Clean title and abstract based on vocab, with minimum of 2 occurences per word
        self._train['text'] = self._train['text'].apply(lambda text: self._clean_text(text))
        self._test['text'] = self._test['text'].apply(lambda text: self._clean_text(text))

    def _create_vocab(self, df):
        """
        Create a vocabulary from a dataframe
        """
        # Add title and abstract text to vocab
        df.apply(lambda row: self._add_text_to_vocab(row['text']), axis=1)

        # print('Vocabulary length: {},\nMost common words:\n{}'.format(len(self.vocab), self.vocab.most_common(50)))

        # Reduce vocabulary length by deleting tokens occuring less than N times
        min_occurrence = 10
        self.vocab_tokens = [k.encode("ascii", errors="ignore").decode() for k, c in self.vocab.items() if
                             c >= min_occurrence]
        # print('Tokens length: {}'.format(len(self.vocab_tokens)))

    def _load_dataset(self, path):
        # load the cleaned dataset
        df = pd.read_excel(path)
        df = df.replace(np.nan, '', regex=True)

        # We combine the title and abstract into a column named 'text'
        df['text'] = df['title'].str.cat(df['abstract'], sep=" ")
        return df[['text', 'label']]

    def _create_tf_datasets(self, buffer_size, batch_size, oversample=None):
        weights = None

        if oversample:
            # Make balanced classes
            weights = [0.5, 0.5]

        # Separate pos and neg training features and labels
        bool_train_labels = self._train['label'].values != 0
        pos_features = self._train[bool_train_labels]['text']
        neg_features = self._train[~bool_train_labels]['text']

        pos_labels = self._train[bool_train_labels]['label']
        neg_labels = self._train[~bool_train_labels]['label']

        # Create a pos and neg training dataset
        pos_ds = self._encoded_dataset(pos_features, pos_labels).shuffle(buffer_size).repeat()
        neg_ds = self._encoded_dataset(neg_features, neg_labels).shuffle(buffer_size).repeat()

        # Create a validation dataset
        test_ds = self._encoded_dataset(self._test['text'], self._test['label'])

        # Oversample the training dataset so it is evenly weighted. Pad both train and val dataset
        train_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=weights)
        self._train_ds = train_ds.padded_batch(batch_size,
                                               padded_shapes=(([self._max_length]), ([]))).prefetch(2)
        self._test_ds = test_ds.padded_batch(batch_size,
                                             padded_shapes=(([self._max_length]), ([]))).prefetch(2)

    def _create_embedding_matrix(self, path='embeddings/glove.6B.100d.txt'):

        # Load the pretrained GloVe embeddings as a Word2Vec model
        tmp_file = get_tmpfile("glove_word2vec.txt")
        _ = glove2word2vec(path, tmp_file)
        self._embedding_model = KeyedVectors.load_word2vec_format(tmp_file)
        word_vectors = self._embedding_model.wv
        # print("Number of word vectors: {}".format(len(word_vectors.vocab)))

        MAX_NB_WORDS = len(word_vectors.vocab)
        self._word_index = {t[0]: i + 1 for i, t in enumerate(self.vocab.most_common(MAX_NB_WORDS))}
        nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab)) + 1

        # we initialize the matrix with random numbers
        wv_matrix = (np.random.rand(nb_words, self.WV_DIM) - 0.5) / 5.0

        for i, word in enumerate(self.vocab):
            if i >= MAX_NB_WORDS:
                continue
            try:
                embedding_vector = word_vectors[word]
                # words not found in embedding index will be all-zeros.
                wv_matrix[i] = embedding_vector
            except KeyError:
                pass
        np.save('weights/' + str(self.WV_DIM) + 'd', wv_matrix)

    def _text_to_clean_tokens(self, text):
        # split into tokens by white space
        tokens = text.split()
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in tokens]
        # convert to lower case
        tokens = [word.lower() for word in tokens]
        return tokens

    def _clean_text(self, text):
        """
        Do some basic text cleaning by creating tokens, remove punctuation and set to lower case. Also remove
        non-alphabetic tokens
        """
        tokens = self._text_to_clean_tokens(text)
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in self.vocab_tokens]
        tokens = ' '.join(tokens)
        return tokens

    def _add_text_to_vocab(self, text):
        """
        Load text and add to vocab
        """
        # basic clean text
        tokens = self._text_to_clean_tokens(text)

        # Additional cleaning
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # update counts
        self.vocab.update(tokens)

    def _encoded_dataset(self, features, labels):
        """
        Make a tf.data.Dataset from pandas dataset
        """
        # integer encode tokens
        encoded = features.apply(lambda row: [self._word_index.get(t, 0) for t in row.split()])
        # pad and truncate text sequences to a maximum lenght
        text_list = pad_sequences(encoded, maxlen=self._max_length, padding='post', truncating='post')

        # get a list from all labels
        label_list = labels.values

        # build the tf.data dataset from respective lists
        text_dataset = tf.data.Dataset.from_tensor_slices(text_list)
        label_dataset = tf.data.Dataset.from_tensor_slices(label_list)

        # zip text and label
        return tf.data.Dataset.zip((text_dataset, label_dataset))


if __name__ == '__main__':
    train_path = 'datasets/train.xlsx'
    test_path = 'datasets/test.xlsx'
    embedding_path = 'embeddings/glove.6B.100d.txt'
    buffer_size = 512
    batch_size = 100

    preprocessor = Preprocessor()
    train_ds, test_ds = preprocessor.get_tf_datasets(train_path, test_path, embedding_path, batch_size, batch_size, show_imbalance=False)
