import argparse
import json
import csv

output_filename = 'data.txt'
def read_data():
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences')
    ap.add_argument('labels', nargs='?')

    args = ap.parse_args()

    sentence_data = open(args.sentences, 'r')
    with open(output_filename, 'w') as fout:
        if args.labels:
            label_data = open(args.labels, 'r')
            for sentence, label in zip(it_sentences(sentence_data), it_labels(label_data)):
                # Tenemos la oración en sentence con su categoría en label
                #print(label, sentence)
                fout.write("__label__"+label+" "+sentence)
                fout.write("\n")
        else:
            for sentence in it_sentences(sentence_data):
                # Tenemos una oración en sentence
                #print(sentence)
                fout.write(sentence)
                fout.write("\n")
    return output_filename

def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']

def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']

read_data()
