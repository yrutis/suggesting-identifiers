from keras.engine.saving import load_model
import logging

def main():
    # load the saved model
    saved_model = load_model('best_model.h5')
    # evaluate the model
    trainX = ''
    trainy = ''
    testX = ''
    testy = ''
    _, train_acc = saved_model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = saved_model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
