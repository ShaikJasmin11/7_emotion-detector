# src/preprocess.py

import pandas as pd

def load_txt_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if ';' in line:
             parts = line.strip().split(';')
             
            if len(parts) == 2:
                    text, label = parts
                    text = text.strip()
                    if text:  # ignore empty strings
                        data.append((text, label))
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df
