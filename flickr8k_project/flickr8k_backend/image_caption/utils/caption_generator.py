"""
Caption generation helper:
- Load caption model and tokenizer
- Generate a caption from extracted image features (simplified greedy decoding)
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_FILE = "data/tokenizer.pkl"
CAPTION_MODEL_FILE = "caption_model.h5"
MAX_LEN = 30

tokenizer = None
caption_model = None

def load_caption_tools():
    global tokenizer, caption_model
    if tokenizer is None:
        tokenizer = pickle.load(open(TOKENIZER_FILE, "rb"))
    if caption_model is None:
        caption_model = load_model(CAPTION_MODEL_FILE)
    return tokenizer, caption_model

def generate_caption(image_feature_vector):
    """
    image_feature_vector: numpy array shape (1, 2048) (features extracted by Inception/ResNet)
    Greedy decode: start token omitted for brevity; token mapping assumed.
    """
    tokenizer, model = load_caption_tools()
    inv_map = {v:k for k,v in tokenizer.word_index.items()}
    # start with empty sequence
    in_text = []
    for i in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([image_feature_vector, seq], verbose=0)
        y_index = np.argmax(yhat)
        word = inv_map.get(y_index, None)
        if word is None:
            break
        in_text.append(word)
        if word == 'endseq':
            break
    caption = " ".join(in_text)
    return caption.replace("endseq", "").strip()