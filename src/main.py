import fasttext
import sys
import os

def main():
    if(not len(sys.argv) == 3):
        print("Te faltan argumentos mostro")
    else:
        train_data, test_data = sys.argv[1], sys.argv[2]
        model = fasttext.train_supervised(train_data)
        print(model.test(test_data))

main()
