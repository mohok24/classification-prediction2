import os
import re
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from docx import Document
import glob

folder_path = r'D:\reports\meow'
reports_data = []

def extract_text_from_docx(docx_file_path):
    doc = Document(docx_file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text.replace("\xa0", " "))
    return text

characters_to_remove = "':,0123456789"
replacement_mapping = {
    "’": " "
}

for file_path in glob.glob(os.path.join(folder_path, "*.docx")):
    report_content = extract_text_from_docx(file_path)
    report_content = ' '.join(report_content)

    for char, replacement in replacement_mapping.items():
        report_content = report_content.replace(char, replacement)

    report_content = re.sub('[' + re.escape(characters_to_remove) + ']', '', report_content)

    results = re.compile(r'RESULTATS(.+?)(?:Conclusion)', re.IGNORECASE | re.DOTALL)

    results_matches = results.findall(report_content)

    if results_matches:
        results_text = ' '.join(results_matches[0].split())

        sentences = sent_tokenize(results_text)
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

        for sentence in tokenized_sentences:
            reports_data.append(sentence)

df = pd.DataFrame({'text': reports_data})
df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessed data saved to preprocessed_data.csv.")
