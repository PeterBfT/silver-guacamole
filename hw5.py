from math import log


class CountVectorizer():
    """Make a word-count matrix"""

    def fit_transform(self, corpus: list = []) -> list | str:
        """Make a matrix of using words from vocabulary of all texts
        in every sentence"""
        self.corpus = corpus
        if not self.checking_input(corpus):
            return 'Error'

        old_matrix = []
        for sentence in corpus:
            old_matrix.append(sentence.lower().split(' '))

        self.vocabulary = self.get_feature_names()
        vocab_dict = {word: 0 for word in self.vocabulary}

        self.matrix = []
        for sentence in old_matrix:
            vocab_dict_copy = vocab_dict.copy()
            for word in sentence:
                if word in vocab_dict_copy.keys():
                    vocab_dict_copy[word] += 1
            self.matrix.append([cnt for cnt in vocab_dict_copy.values()])

        return self.matrix

    def get_feature_names(self) -> list | str:
        """Make a vocabulary from all texts"""
        if not self.checking_input(self.corpus):
            return 'Error'

        all_words = ' '.join(self.corpus).lower().split(' ')

        vocabulary = []
        for word in all_words:
            if word not in vocabulary:
                vocabulary.append(word)

        return vocabulary

    @staticmethod
    def checking_input(array: list) -> bool:
        """Checks if input is correct"""
        if not isinstance(array, list):
            return False
        for word in array:
            if not isinstance(word, str):
                return False
        return True


class TfidTransformer():

    def fit_transform(self, matrix: list) -> list | str:
        """Make a tf-idf matrix from matrix of words-counters"""
        if not self.checking_input(matrix):
            return 'Error'

        tf_matrix = self.tf_transform(matrix)
        idf_matrix = self.idf_transform(matrix)

        tfidf_matrix = []
        for sent in tf_matrix:
            tfidf_array = []
            for word1, word2 in zip(sent, idf_matrix):
                tfidf_array.append(round(word1 * word2, 3))
            tfidf_matrix.append(tfidf_array)
        return tfidf_matrix

    def tf_transform(self, old_matrix: list) -> list:
        """Make a term-frequency matrix from matrix of words-counters"""
        tf_matrix = []
        for sent in old_matrix:
            matrix_of_tf = []
            sum_words = sum(sent)
            for word_cnt in sent:
                tf = word_cnt / sum_words
                matrix_of_tf.append(round(tf, 3))
            tf_matrix.append(matrix_of_tf)
        return tf_matrix

    def idf_transform(self, old_matrix) -> list:
        """Make an inverse term-frequency matrix
        from matrix of words-counters"""
        sent_cnter = len(old_matrix)
        cnt_words = [0] * len(old_matrix[0])
        for sent in old_matrix:
            for i, val in enumerate(sent):
                if val > 1:
                    val = 1
                cnt_words[i] += val
        idf_matrix = []
        for val in cnt_words:
            idf = log((sent_cnter + 1) / (val + 1)) + 1
            idf_matrix.append(round(idf, 3))
        return idf_matrix

    @staticmethod
    def checking_input(matrix: list) -> bool:
        """Checks if input is correct"""
        if not isinstance(matrix, list):
            return False
        for sent in matrix:
            if not isinstance(sent, list):
                return False
            for val in sent:
                if not isinstance(val, int):
                    return False
        return True


class TfidVectorizer(CountVectorizer):

    def __init__(self) -> None:
        super().__init__()

    def fit_transform(self, old_matrix: list) -> list | str:
        """Make a term-frequency matrix from list of sentences"""
        matrix = super().fit_transform(old_matrix)
        return TfidTransformer().fit_transform(matrix)


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = TfidVectorizer()
    tdidf_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(tdidf_matrix)
