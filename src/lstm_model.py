import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_classifier import LSTMClassifier

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
def get_data(data):
    data_list = []
    with open(data, 'r') as sentences:
        for line in sentences:
            data_list.append((get_sentence(line), get_label(line)))
    return data_list

def get_label(line):
    return line.split()[0].replace('__label__', '')

def get_sentence(line):
    line_split = line.split();
    res = "";
    for val in range(1, len(line.split())):
        res += line_split[val] + " ";
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
    return torch.LongTensor([label_to_ix[label]])

def get_label_by_item(item):
    for label, value in label_to_ix.items():
        if(value == item):
            return label
    return None

''' Creación del modelo '''
### Defino la cantidad de palabras y la cantidad de labels
label_to_ix = { "neutral": 0, "contradiction": 1, "entailment": 2 }
word_to_ix = create_map([DEV_SENTENCES, TRAIN_SENTENCES, TEST_SENTENCES])
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(label_to_ix)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Creo mi modelo, defino la loss function, y la función de optimización
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LABELS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# TODO mover esto de aca
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w], seq))
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

'''Entrenamiento'''
# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable (NOTA DE MARISCO: tarda algunos minutos cada vuelta).
data = get_data(TRAIN_DATA)
for epoch in range(10):
    running_loss = 0.0
    i = 0
    for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()
        model.hidden = model.init_hidden()

        # Step 2. Make our BOW vector and also we must wrap the target in a Variable
        # as an integer.  For example, if the target is SPANISH, then we wrap the integer
        # 0.  The loss function then knows that the 0th element of the log probabilities is
        # the log probability corresponding to SPANISH
        boke = instance.split()
        bow_vec = prepare_sequence(boke, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        # print statistics
        i += 1
        running_loss += loss.item()
        if i % 2000 == 1999:# print every 200000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

'''Predicción'''
test_data = get_data(DEV_DATA)
counter = 0
ok = 0
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    _, predicted = torch.max(log_probs, 1)
    if(get_label_by_item(predicted.item()) == label):
        ok += 1
    counter += 1
print("Resultado: {} ".format(((100*ok)/counter)/100))
