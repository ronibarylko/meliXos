import fasttext
import sys
import os
import argparse

'''
No hay mucha magia (o al revés: hay demasiada).
Usamos la librería de fasttext para entrenar nuestro modelo, y luego appendeamos las predicciones con el valor __label__ pues esa es la salida esperada
'''

def train_and_test():
    model = train(args.train_data)
    print(test(model, args.test_data))

def train_and_predict():
    model = train(args.train_data)
    print(predict(model, args.test_data))

def train(train_data):
    return fasttext.train_supervised(train_data, epoch=5, wordNgrams=2, verbose=2)

def test(model, test_data):
    return model.test(test_data)

def predict(model, predict_file):
    with open(predict_file, 'r') as sentences, open('result.txt', 'w') as fout:
        for line in sentences:
            result = model.predict(line.replace('\n', ''))
            get_result_label(result)
            fout.write(get_result_label(result))
            fout.write('\n')

def get_result_label(result):
    return result[0][0]

ap = argparse.ArgumentParser()
ap.add_argument('train_data')
ap.add_argument('test_data')
ap.add_argument('function')

args = ap.parse_args()

if(args.function == 'test'):
    train_and_test()
else:
    if(args.function == 'predict'):
        train_and_predict()
    else:
        raise Exception('{} is not test or predict'.format(args.function))
