import numpy as np

class Vocabulary():

    @staticmethod
    def revert_back(tokenizer, sequence):
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

        if type(sequence) is list:
            words = [reverse_word_map.get(letter) for letter in sequence]
            return (words)
        if type(sequence) is int:
            word = reverse_word_map.get(sequence)
            return word
        if type(sequence) is np.ndarray:
            words = [reverse_word_map.get(letter) for letter in sequence]
            words = words
            return words
        else:
            raise Exception("Cannot convert type is {} and is not list / int / np.ndarray".format(type(sequence)))




