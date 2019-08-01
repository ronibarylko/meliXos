import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_classifier import LSTMClassifier
from torch.utils.data import DataLoader, TensorDataset
from custom_dataset import CustomDataset
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('function')

args = ap.parse_args()

'''
Modelo cheto, vamos por ti
Primero y principal, consideremos que estoy armando de entrada el modelo para el BoW. Quizas después necesitamos distribuir en clases y bla. Ojala que no. Questo e Boca.
'''

DEV_SENTENCES = "dev_sentences.txt"
TRAIN_SENTENCES = "train_sentences.txt"
TEST_SENTENCES = "test_sentences.txt"

DEV_DATA = "dev_data.txt"
TRAIN_DATA = "train_data.txt"

'''Funciones'''
# Función que levanta el archivo data y lo transforma en una lista de (sentence, label)
def get_data_splitted(data):
    instances = []
    labels = []
    with open(data, 'r') as sentences:
        for line in sentences:
            instances.append(get_sentence_splitted(line))
            labels.append(get_label(line))
    return instances, labels

def get_instances_to_predict(file, batch_size):
    sentences = []
    with open(file, 'r') as infile:
        for line in infile:
            sentences.append(line)


def get_label(line):
    return line.split()[0].replace('__label__', '')

def get_sentence_splitted(line):
    line_split = line.split();
    res = []
    for val in range(1, len(line_split)):
        res.append(line_split[val])
    return res;

### Función que toma un jsonl y agrega las palabras a mi mapa de word_to_integer
def add_words_to_map(sentences, word_to_ix):
    for sentence in sentences:
        for word in sentence.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

### Función que recibe la lista de archivos txt, convierte cada uno en una lista de oraciones de Python y se encarga de llamar a add_words_to_map
def create_map(txt_list):
    word_to_ix = {}
    for input_file in txt_list:
        with open(input_file, 'r') as infile:
            sentences = []
            for line in infile:
                sentences.append(line)
            word_to_ix = add_words_to_map(sentences, word_to_ix)
    return word_to_ix

# Función que crea un vector contando la cantidad de apariciones de las palabras en una oración.
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix)) # Vector de ceros
    for word in sentence.split():
        vec[word_to_ix[word]] += 1 # Por cada aparición de una palabra, le sumo uno
    return vec.view(1, -1) # Vector de tamaño 1 x n, donde n es inferido por el tamaño de palabras

# Función que wrappea la variable en un tensor. Básicamente, le pasas la lista de labels y tu label en particular, y te devuelve un tensor con el valor 0, 1 ó 2 adentro.
def make_target(label, label_to_ix):
    return label_to_ix[label]

def get_label_by_item(item):
    for label, value in label_to_ix.items():
        if(value == item):
            return label
    return None

def calculate_prediction_rate(predicted, label_batch, counter, ok):
    for instance,label in zip(predicted, label_batch):
        if(instance.item() == label.item()):
            ok += 1
        counter += 1
    return counter, ok

''' Creación del modelo '''
### Defino la cantidad de palabras y la cantidad de labels
label_to_ix = { "neutral": 0, "contradiction": 1, "entailment": 2 }
word_to_ix = create_map([DEV_SENTENCES, TRAIN_SENTENCES, TEST_SENTENCES])
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(label_to_ix)
EMBEDDING_DIM = 300
HIDDEN_DIM = 150
BATCH_SIZE = 200
EPOCH_SIZE = 5

# Creo mi modelo, defino la loss function, y la función de optimización
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LABELS, BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

# TODO mover esto de aca
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w], seq))
    return idxs

def get_result_label(result, label_to_ix):
    for label, number in label_to_ix.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if result == number:
            return label

def get_tensor_data(data_inst, data_lab, word_to_ix, label_to_ix, use_labels=True):
    instances = []
    labels = []
    for instance, label in zip(data_inst, data_lab):
        instances.append(prepare_sequence(instance, word_to_ix))
        if(use_labels):
            labels.append(make_target(label, label_to_ix))
        else:
            labels.append(0)
    return instances, labels

'''Entrenamiento'''
# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable (NOTA DE MARISCO: tarda algunos minutos cada vuelta).
instances, labels = get_data_splitted(DEV_DATA)
instances, labels = get_tensor_data(instances, labels, word_to_ix, label_to_ix)

def get_max_length(x):
    return len(max(x, key=len))

def pad_sequence(seq):
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it
    return [_pad(it, get_max_length(seq)) for it in seq]

def custom_collate(batch):
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if isinstance(samples[0], int):
            lst.append(torch.LongTensor(samples))
        elif isinstance(samples[0], float):
            lst.append(torch.DoubleTensor(samples))
        elif isinstance(samples[0], list):
            lst.append(torch.LongTensor(pad_sequence(samples)))
    return lst

tensor_data = CustomDataset(instances, labels)
train_loader = DataLoader(dataset=tensor_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, drop_last=True) #TODO shuffle? drop_last es una verga
counter = 0
ok = 0
for epoch in range(EPOCH_SIZE):
    running_loss = 0.0
    i = 0
    for instance_batch, label_batch in train_loader:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()
        model.hidden = model.init_hidden()
        instance_batch = instance_batch.transpose(0,1)

        # Step 2. Make our BOW vector and also we must wrap the target in a Variable
        # as an integer.  For example, if the target is SPANISH, then we wrap the integer
        # 0.  The loss function then knows that the 0th element of the log probabilities is
        # the log probability corresponding to SPANISH

        # Step 3. Run our forward pass
        log_probs = model(instance_batch)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(log_probs, label_batch)
        loss.backward()
        optimizer.step()
        # print statistics
        i += 1
        running_loss += loss.item()
        _, predicted = torch.max(log_probs, 1)
        counter, ok = calculate_prediction_rate(predicted, label_batch, counter, ok)
        melixiTos = 100
        if i % melixiTos == (melixiTos-1):# print every 200000 mini-batches
            print('[%d, %5d] loss: %.3f, predicted: %.3f' % (epoch + 1, (i + 1)*BATCH_SIZE, running_loss / (melixiTos), ((100*ok)/counter)/100))
            counter = 0
            ok = 0
            running_loss = 0.0

'''Predicción'''
if(args.function == 'test'):
    instances, labels = get_data_splitted(DEV_DATA)
    instances, labels = get_tensor_data(instances, labels, word_to_ix, label_to_ix)
    tensor_data = CustomDataset(instances, labels)
    train_loader = DataLoader(dataset=tensor_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, drop_last=True)
    counter = 0
    ok = 0
    for instance_batch, label_batch in train_loader:
        instance_batch = instance_batch.transpose(0,1)
        log_probs = model(instance_batch)
        _, predicted = torch.max(log_probs, 1)
        counter, ok = calculate_prediction_rate(predicted, label_batch, counter, ok)
    print("Resultado: {} ".format(((100*ok)/counter)/100))
else:
    if(args.function == 'predict'):
        with open('result.txt', 'w') as fout:
            instances, labels = get_data_splitted(TEST_SENTENCES)
            instances, labels = get_tensor_data(instances, labels, word_to_ix, label_to_ix, use_labels=False)
            tensor_data = CustomDataset(instances, labels)
            train_loader = DataLoader(dataset=tensor_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, drop_last=True)
            for instance_batch, _ in train_loader:
                instance_batch = instance_batch.transpose(0,1)
                log_probs = model(instance_batch)
                _, predicted = torch.max(log_probs, 1)
                for label_prediction in predicted:
                    fout.write(get_result_label(label_prediction, label_to_ix))
                    fout.write('\n')
                raise Exception('NO dropees mogolico')
    else:
        raise Exception('{} is not test or predict'.format(args.function))
