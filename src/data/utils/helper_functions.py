import logging
import re
from collections import Counter
from itertools import dropwhile

import pandas as pd
import numpy as np


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

def remove_bad_naming_methods(df):
    '''

    :param df: before bad naming methods have been removed
    :return: df: after bad naming methods have been removed
    '''

    logger = logging.getLogger(__name__)
    i = 0
    length_before_removing = df.shape[0]
    logger.info("before removing get0 until get1000 {}".format(df.shape[0]))
    while i < 1000:
        #removed_methods = df[df['methodName'] == 'get'+str(i)]
        #if removed_methods.shape[0] > 0:
            #logger.info("this amount {} for the method get{} have just been removed".format(len(removed_methods['methodName'].tolist()), str(i)))
        df = df[df['methodName'] != 'get'+str(i)]
        i += 1
    logger.info("after removing get0 until get1000 {}, difference {}".format(df.shape[0], length_before_removing-df.shape[0]))
    return df

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


    training = list(c.keys())
    training_string = [" ".join(training)]
    return training_string


def get_first_x_elem(elem_list, window_size):
    if len(elem_list) <= window_size:
        return elem_list
    else:
        return elem_list[:window_size]


def remove_some_unknowns(trainX, trainY, valX, valY, remove_train=0, remove_val=0):
    '''
    :param trainX:
    :param trainY:
    :param valX:
    :param valY:
    :param remove_train: flag if some unknowns from training should be removed
    :param remove_val: flag if some unknowns from validation should be removed
    :return: trainX, trainY, valX, valY, percentage unknowns train, percentage unknowns val
    '''

    logger = logging.getLogger(__name__)
    logger.info("this is remove_train {}, this is remove_val {}".format(remove_train, remove_val))

    train_df = pd.DataFrame({'trainY': trainY, 'trainX': list(trainX)})
    logger.info(train_df.head())
    perc_unk_train = (len(train_df[(train_df['trainY'] == 1)])) / (len(train_df.index))
    logger.info("This is the percentage of UNK in Training before removal {}".format(perc_unk_train))


    if remove_train > 0 and remove_train < 1:
        train_df = train_df.drop(train_df[train_df['trainY'] == 1].sample(frac=remove_train).index)
        perc_unk_train = (len(train_df[(train_df['trainY'] == 1)])) / (len(train_df.index))
        logger.info("This is the Percentage of UNK during training {}".format(perc_unk_train))

    elif remove_train == 1:
        train_df = train_df.drop(train_df[train_df['trainY'] == 1].index)
        perc_unk_train = len(train_df[(train_df['trainY'] == 1)]) / (len(train_df.index))
        logger.info("This is the Percentage of UNK during training {}".format(perc_unk_train))

    trainX = np.array(train_df['trainX'].values.tolist())
    trainY = train_df['trainY'].values


    val_df = pd.DataFrame({'valY': valY, 'valX': list(valX)})

    perc_unk_val = len(val_df[(val_df['valY'] == 1)]) / len(val_df.index)
    logger.info("This is the percentage of UNK in Validation before removal {}".format(perc_unk_val))

    if remove_val > 0 and remove_val < 1:
        val_df = val_df.drop(val_df[val_df['valY'] == 1].sample(frac=remove_val).index)
        perc_unk_val = (len(val_df[(val_df['valY'] == 1)])) / (len(val_df.index))
        logger.info("This is the percentage of UNK in Validation after removal {}".format(perc_unk_val))
    elif remove_val == 1:
        val_df = val_df.drop(val_df[val_df['valY'] == 1].index)
        perc_unk_val = (len(val_df[(val_df['valY'] == 1)])) / (len(val_df.index))
        logger.info("This is the percentage of UNK in Validation after removal {}".format(perc_unk_val))


    valX = np.array(val_df['valX'].values.tolist())
    valY = val_df['valY'].values


    return trainX, trainY, valX, valY, perc_unk_train, perc_unk_val


def remove_some_unknowns_test(testX, testY, remove_test=0):
    '''
    :param testX:
    :param testY:
    :param remove_test: percentage between 0 and 1
    :return: testX, testY, perc unk test
    '''

    logger = logging.getLogger(__name__)


    test_df = pd.DataFrame({'testY': testY, 'testX': list(testX)})

    perc_unk_test = len(test_df[(test_df['testY'] == 1)]) / len(test_df.index)
    logger.info("This is the percentage of UNK in Test before removal {}".format(perc_unk_test))

    if remove_test > 0 and remove_test < 1:
        test_df = test_df.drop(test_df[test_df['testY'] == 1].sample(frac=remove_test).index)
        perc_unk_test = (len(test_df[(test_df['testY'] == 1)])) / (len(test_df.index))
        logger.info("This is the percentage of UNK in Test after removal {}".format(perc_unk_test))
    elif remove_test == 1:
        test_df = test_df.drop(test_df[test_df['testY'] == 1].index)
        perc_unk_test = (len(test_df[(test_df['testY'] == 1)])) / (len(test_df.index))
        logger.info("This is the percentage of UNK in Test after removal {}".format(perc_unk_test))


    testX = np.array(test_df['testX'].values.tolist())
    testY = test_df['testY'].values


    return testX, testY, perc_unk_test


def getFirstElem(x):
    logger = logging.getLogger(__name__)
    try:
        return x[0]
    except IndexError:
        #if for some reason there is no method name (due to weird namings...) -> map it to unknown
        logger.info(x)
        return 1