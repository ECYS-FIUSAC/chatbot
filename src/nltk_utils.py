import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from langdetect import detect
from model import NeuralNet



# Inicializar el stemmer para español y el lematizador para inglés
stemmer_es = SnowballStemmer("spanish")
lemmatizer_en = WordNetLemmatizer()

# Funciones auxiliares para procesar
def detectar_idioma(oracion):
    try:
        idioma = detect(oracion)
        return "spanish" if idioma == "es" else "english"
    except:
        return "spanish"  # Por defecto español

def dividir_en_palabras(oracion, idioma="spanish"):
    oracion = oracion.lower()
    oracion = re.sub(r'[^\w\s]', '', oracion)  # Eliminar signos de puntuación
    tokens = nltk.word_tokenize(oracion, language=idioma)
    return tokens

def obtener_pos_etiqueta(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def obtener_raiz(palabra, idioma="spanish"):
    excepciones = ["suma", "resta", "multiplicacion", "division", "+", "-", "*", "/"]
    if palabra in excepciones or palabra.isdigit():
        return palabra
    if idioma == "spanish":
        raiz = stemmer_es.stem(palabra)
        #print(f"[DEBUG] Palabra original: {palabra}, Raíz en español: {raiz}")
        return raiz
    elif idioma == "english":
        pos_tag = nltk.pos_tag([palabra])[0][1]
        wordnet_tag = obtener_pos_etiqueta(pos_tag)
        raiz = lemmatizer_en.lemmatize(palabra, pos=wordnet_tag)
        #print(f"[DEBUG] Palabra original: {palabra}, Raíz en inglés: {raiz}")
        return raiz

def vector_bolsa_de_palabras(oracion_tokenizada, palabras_conocidas, idioma="spanish"):
    palabras_raiz = [obtener_raiz(palabra, idioma) for palabra in oracion_tokenizada]
    palabras_no_encontradas = [palabra for palabra in palabras_raiz if palabra not in palabras_conocidas]
    #print(f"[DEBUG] Palabras raíz generadas: {palabras_raiz}")
    #print(f"[DEBUG] Palabras raíz no encontradas en all_words: {palabras_no_encontradas}")

    vector = np.zeros(len(palabras_conocidas), dtype=np.float32)
    for idx, palabra in enumerate(palabras_conocidas):
        if palabra in palabras_raiz:
            vector[idx] = 1
    return vector
