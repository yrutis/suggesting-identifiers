import numpy as np
from matplotlib import pyplot as plt

import matplotlib


# function for plotting the attention weights
def plot_attention():
    attention = np.array([[7.78877574e-10, 4.04739769e-10, 6.65854022e-05, 1.63362725e-04,
                           2.85054208e-04, 8.50252633e-04, 4.58042100e-02],
                          [9.23501700e-02, 5.69618285e-01, 1.80586591e-01, 9.78111699e-02,
                           2.71992851e-02, 9.59911197e-03, 2.54837354e-03],
                          [3.49277966e-02, 2.07805950e-02, 6.57206476e-01, 2.21483693e-01,
                           4.95323166e-02, 8.13153014e-03, 6.41921011e-04],
                          [2.18881600e-04, 2.74454615e-05, 2.53675003e-02, 8.90285194e-01,
                           7.41584152e-02, 9.66714323e-03, 1.09222419e-04],
                          [2.72368439e-08, 1.10686379e-07, 3.47205896e-05, 5.99798467e-03,
                           7.39551187e-01, 2.27898076e-01, 2.52859760e-02],
                          [4.79354547e-08, 9.38888718e-08, 2.94979673e-05, 1.02036993e-03,
                           1.76065769e-02, 7.77017593e-01, 2.03639299e-01],
                          [1.89887821e-08, 6.92338844e-08, 3.41121267e-05, 4.09495260e-04,
                           5.14712185e-03, 5.07713616e-01, 4.84622657e-01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    sentence = ['<start>', 'hace', 'mucho', 'frio', 'aqui', '.', '<end>']
    predicted_sentence = ['it', 's', 'too', 'cold', 'here', '.', '<end>', '']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    print(attention)
    print(len(sentence), len(predicted_sentence))
    print(sentence, predicted_sentence)
    print(matplotlib.__version__)
    print([''] + sentence)
    print([''] + predicted_sentence)

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()
    plt


plot_attention()