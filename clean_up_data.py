import csv
import os
import re
import shutil
from pathlib import Path
from random import sample

import nltk
from nltk.corpus import stopwords, wordnet

stop_words_list = stopwords.words('english')
stop_words_list.extend(['could', 'may', 'n\'t', 'pls', 'please', 'hmm', 'whao', 'wowo',
                        'wow', 'im', 'omg', '\'s', 'lol', '\'m', 'pm', 'am',
                        # 'month', 'year'
                        ])
original_file_path = 'original/train.csv'
temp_data_file_path = 'cleaned_data/temp_all.csv'
cleaned_data_file_path = 'all.csv'
positive_data_file_path = 'cleaned_data/positive'
negative_data_file_path = 'cleaned_data/negative'
SPLIT_RATIO = 0.8


def generate_csv(should_generate_file_system=False):
    open(temp_data_file_path, 'w').close()
    open(cleaned_data_file_path, 'w').close()
    with open(original_file_path, newline='') as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=',')
        for row in spam_reader:
            if row[4] == 'target':
                continue
            content = row[3]
            content = content.replace('\n\n', '. ')
            sent_text = nltk.sent_tokenize(content)
            cleaned_sentences = []
            for sentence in sent_text:
                sentence = re.sub('http(s?)://[./a-zA-Z0-9]+', '', sentence)
                sentence = re.sub('[^a-zA-Z0-9 \'@]+', ' ', sentence)
                sentence = sentence + "."
                # february
                sentence = re.sub('(18|20)\\d{2}\\D', 'epoch ', sentence)
                sentence = re.sub(''
                                  '((J| [Jj])[Aa][Nn](uary|UARY)?\\W)|'
                                  '((F| [Ff])[Ee][Bb](ruary|RUARY)?\\W)|'
                                  '((M| [Mm])[Aa][Rr](ch|CH)?\\W)|'
                                  '((A| [Aa])[Pp][Rr](il|IL)?\\W)|'
                                  # Not sure how to deal with May
                                  '((J| [Jj])[Uu][Nn]([Ee])?\\W)|'
                                  '((J| [Jj])[Uu][Ll]([Yy])?\\W)|'
                                  '((A| [Aa])[Uu][Gg](ust|UST)?\\W)|'
                                  '((S| [Ss])[Ee][Pp](tember|TEMBER)?\\W)|'
                                  '((O| [Oo])[Cc][Tt](ober|OBER)?\\W)|'
                                  '((N| [Nn])[Oo][Vv](ember|EMBER)?\\W)|'
                                  '((D| [Dd])[Ee][Cc](ember|EMBER)?\\W)'
                                  '', ' january ', sentence)
                tokenized_text = nltk.word_tokenize(sentence)
                word_tokens = []
                should_skip_next_word = False
                for word in tokenized_text:
                    word = word.lower()
                    if should_skip_next_word:
                        should_skip_next_word = False
                        continue
                    if word.startswith('@'):
                        should_skip_next_word = True
                        continue
                    if not word.isdigit():
                        if word not in stop_words_list:
                            if wordnet.synsets(word):
                                word_tokens.append(word)
                cleaned_sentence = " ".join(word_tokens)
                cleaned_sentence = cleaned_sentence.replace(' .', '.')
                if cleaned_sentence != ".":
                    cleaned_sentences.append(cleaned_sentence)
            content = " ".join(cleaned_sentences)
            content = content.strip()
            if len(content) > 0:
                result = [row[4], content]
                with open(temp_data_file_path, 'a') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(result)
    with open(temp_data_file_path, 'r') as in_file, open(cleaned_data_file_path, 'w') as out_file:
        seen = set()
        for line in in_file:
            if line[2:] in seen:
                # print(line)
                continue
            seen.add(line[2:])
            out_file.write(line)
    if should_generate_file_system:
        os.remove(temp_data_file_path)
        generate_file_system()


def generate_file_system():
    shutil.rmtree(positive_data_file_path, ignore_errors=True)
    shutil.rmtree(negative_data_file_path, ignore_errors=True)
    Path(positive_data_file_path).mkdir(parents=True, exist_ok=True)
    Path(negative_data_file_path).mkdir(parents=True, exist_ok=True)
    with open(cleaned_data_file_path, newline='') as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=',')
        positive_count = 0
        negative_count = 0
        for row in spam_reader:
            if row[0] == '1':
                with open(os.path.join(positive_data_file_path, 'pos_' + str(positive_count) + '.txt'), 'w') as file:
                    file.write(row[1])
                positive_count = positive_count + 1
            else:
                with open(os.path.join(negative_data_file_path, 'neg_' + str(negative_count) + '.txt'), 'w') as file:
                    file.write(row[1])
                negative_count = negative_count + 1
    if positive_count > negative_count:
        num_of_files_to_remove = positive_count - negative_count
        randomly_remove_x_files(positive_data_file_path, num_of_files_to_remove)
    if negative_count > positive_count:
        num_of_files_to_remove = negative_count - positive_count
        randomly_remove_x_files(negative_data_file_path, num_of_files_to_remove)


def randomly_remove_x_files(path, num_of_files):
    files = os.listdir(path)
    for file in sample(files, num_of_files):
        os.remove(os.path.join(path, file))
