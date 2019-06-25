from abc import abstractmethod

class AbstractTrainSubtoken(object):


    @abstractmethod
    def predict(self, tokenizer, input_seq, k, return_top_n):
        return