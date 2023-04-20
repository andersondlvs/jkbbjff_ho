import csv
import spacy
import nltk
import sklearn
import tensorflow as tf
import string
from nltk.corpus import stopwords
import pt_core_news_sm

# Carregar o modelo de linguagem em português
nlp = spacy.load("pt_core_news_sm")
nlp = pt_core_news_sm.load()

# Definir as funções para pré-processar e processar o texto
def preprocess_text(text):
    # Remover pontuação e transformar em minúsculas
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()

    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    # Juntar as palavras novamente em uma string
    processed_text = ' '.join(filtered_words)

    return processed_text

def process_text(text):
    doc = nlp(text)

    # Extrair entidades nomeadas e relacionamentos
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    relations = []
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'nsubj':
            subject = chunk.text
            for token in chunk.root.children:
                if token.dep_ == 'dobj':
                    object = token.text
                    relations.append((subject, object))

    # Retornar um dicionário com as entidades e relações extraídas
    processed_text = {'entities': entities, 'relations': relations}

    return processed_text

# Carregar o arquivo de dados de treinamento
x_train = []
y_train = []

with open('dados.txt', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        question = preprocess_text(row[0])
        x_train.append(process_text(question))
        y_train.append(row[1])

# Definir as camadas de processamento de texto e aprendizado de máquina
vocab_size = 10000
embedding_dim = 16
max_length = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar e treinar o modelo com os dados de treinamento
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Implementar a interface de usuário via terminal
while True:
    user_input = input("Digite sua pergunta ou comando: ")

    # pré-processar e processar o texto do usuário
    processed_input = preprocess_text(user_input)
    processed_input = process_text(processed_input)

    # usar o modelo para gerar uma resposta ou código HTML
    response = model.predict(processed_input)

    # imprimir a resposta ou código HTML gerado
    print(response)
