import re
from collections import Counter
from itertools import dropwhile

def removeOptional(method_body):
    '''
    :param method_body: methodbody
    :return: removed first 10 chars from each line
    remove "optional[" (when content) or "optional." when empty in the beginning of each line
    '''

    remove_optional = str(method_body)[9:-1]
    return remove_optional


def replace_string_values(method_body):
    '''

    :param method_body: method body
    :return: string value like: string = "I need help" --> string = STRINGVALUE
    '''

    method_body = re.sub(r'"([^"]*)"', "STRINGVALUE", method_body)
    return method_body



def turn_strings_to_list(method_body):
    '''

    :param method_body: methodbody
    :return: list: list with each element a word in function
    '''

    method_body = re.split('(\W)', method_body)
    return method_body


def delete_certain_strings(method_body):
    '''

    :param method_body: methodbody
    :return: list: with empty, only space, \r, \n, empty elements deleted
    '''

    method_body = list(filter(None, method_body))
    method_body = list(filter(lambda x: x != " ", method_body))
    method_body = list(filter(lambda x: x != "\r", method_body))
    method_body = list(filter(lambda x: x != "\n", method_body))
    method_body = list(filter(lambda x: x != "empty", method_body))
    return method_body


def compute_col_length(method_body):
    '''

    :param method_body: methodbody
    :return: length for each methodybody
    '''

    length = len(method_body)
    return length


def clean_from_function_structure(method_body):
    '''
    :param method_body: method_body
    :return: returns a cleaned from function structure like ();. form of the method body
    '''
    method_body = list(filter(lambda x: x not in "()'{},.;\"", method_body))
    return method_body


def turn_all_to_lower(value):
    '''

    :param value: string or list of strings

    :return: turns all to lower
    '''

    value = list(map(lambda x: x.lower(), value))
    return value


def split_params(params):
    '''

    :param params: list of strings like ["word word", ...]
    :return: list of strings like ["word", "word",...]
    '''

    concat_list = []
    for x in params:
        z = x.split()
        for elem in z:
            concat_list.append(elem)

    return concat_list

def get_into_tokenizer_format(method_body):
    '''

    :param method_body: methodbody
    :return: transforms methodbody ["word", "word", ...] in format ["word word ..."]
    '''

    method_body = " ".join(method_body)
    return method_body

def get_training_vocab(method_list, is_for_x):
    '''

    :param method_list: method body list or method names
    :param is_for_x: either x or y
    :return: get words that appear more often than 3x, return in right format for fit on text keras
    '''

    if is_for_x:
        flat_list = [item for sublist in method_list for item in sublist]
        #TODO flat_list = flat_list[0: max seq]
    else:
        flat_list = method_list

    c = Counter(flat_list)

    # delete words that appear less than ...
    for key, count in dropwhile(lambda key_count: key_count[1] >= 3, c.most_common()):
        del c[key]

    print("outputting {}".format(len(c)))

    training = list(c.keys())
    training_string = [" ".join(training)]
    return training_string


