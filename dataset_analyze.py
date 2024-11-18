import os
import re
import nltk
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

# Make sure to download necessary NLTK resources if not already done
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Create a directory to save plots
plot_dir = './plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Paths to the dataset folders
training_data_folder = "./data/training_data_16_October_release/EN/raw-documents"
annotation_file = "./data/training_data_16_October_release/EN/subtask-2-annotations.txt"
dev_data_folder = "./data/dev-documents_25_October/EN/raw-documents"

# Load annotation file for analysis
annotations = pd.read_csv(annotation_file, sep='\t', header=None, names=["File", "Annotation1", "Annotation2"])

# Step 1: Load the Training Data
training_files = os.listdir(training_data_folder)
training_texts = {}

all_text = ""  # Initialize all_text to concatenate all documents

for file in training_files:
    with open(os.path.join(training_data_folder, file), 'r', encoding='utf-8') as f:
        content = f.read()
        training_texts[file] = content
        all_text += content + " "  # Concatenate all documents

# Step 2: Basic Text Analysis
# 2.1: Average Sentence Length and Word Count
sentence_lengths = []
word_count = 0
vocabulary = set()

# Variables for metrics including stopwords
word_count_including_stopwords = 0
vocabulary_including_stopwords = set()

# Variables for metrics including everything
word_count_including_everything = 0
vocabulary_including_everything = set()

stop_words = set(stopwords.words('english'))
punctuation = re.compile(r'[\W_]+')

filtered_all_words = []  # Define a list to store all filtered words from all documents

for text in training_texts.values():
    # Tokenize without any cleaning (including punctuation and symbols)
    all_tokens = word_tokenize(text)
    word_count_including_everything += len(all_tokens)
    vocabulary_including_everything.update(all_tokens)

    # Removing punctuation and converting to lowercase
    cleaned_text = punctuation.sub(' ', text).lower()
    words = word_tokenize(cleaned_text)

    # Word count and vocabulary including stopwords
    word_count_including_stopwords += len(words)
    vocabulary_including_stopwords.update(words)

    # Filtering out stopwords
    filtered_words = [word for word in words if word not in stop_words]
    filtered_all_words.extend(filtered_words)

    # Word count and vocabulary excluding stopwords
    word_count += len(filtered_words)
    vocabulary.update(filtered_words)

    # Calculate sentence length
    sentences = re.split(r'[.!?]', cleaned_text)  # Split text into sentences
    sentence_lengths.extend([len(sentence.split()) for sentence in sentences if len(sentence.strip()) > 0])

# Calculating average sentence length
average_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
unique_vocab_size = len(vocabulary)
unique_vocab_size_including_stopwords = len(vocabulary_including_stopwords)
unique_vocab_size_including_everything = len(vocabulary_including_everything)

# Printing Metrics
print(f"Average Sentence Length: {average_sentence_length:.2f} words")
print(f"Total Word Count (excluding stopwords and punctuation): {word_count}")
print(f"Vocabulary Size (excluding stopwords and punctuation): {unique_vocab_size}")
print(f"Total Word Count (including stopwords): {word_count_including_stopwords}")
print(f"Total Vocabulary Size (including stopwords): {unique_vocab_size_including_stopwords}")
print(f"Total Word Count (including everything): {word_count_including_everything}")
print(f"Total Vocabulary Size (including everything): {unique_vocab_size_including_everything}")

# Step 3: POS Tagging
# Get parts of speech tagging for a sample cleaned text
sample_text = list(training_texts.values())[0]
cleaned_sample_text = punctuation.sub(' ', sample_text).lower()
sample_tokens = word_tokenize(cleaned_sample_text)
filtered_sample_tokens = [word for word in sample_tokens if word not in stop_words]
pos_tags = pos_tag(filtered_sample_tokens)

# POS Tag Description Mapping (Penn Treebank POS Tag Set)
pos_tag_map = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb'
}

# Count frequency of POS tags
pos_counts = Counter(tag for word, tag in pos_tags)

# Sort POS counts by frequency
sorted_pos_counts = pos_counts.most_common()

# Extract the sorted POS tags and their descriptions
sorted_pos_tags, sorted_counts = zip(*sorted_pos_counts)
sorted_pos_descriptions = [pos_tag_map.get(tag, "Unknown") for tag in sorted_pos_tags]

# Step 4: Visualization
# 4.1: Word Cloud of Most Common Words (excluding stopwords and punctuation)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_all_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Training Data (excluding punctuation, articles, conjunctions)')
plt.savefig(f"{plot_dir}/word_cloud_filtered.png")
plt.show()

# 4.2: Parts of Speech Frequency Plot (sorted by frequency)
plt.figure(figsize=(14, 6))
plt.bar(sorted_pos_descriptions, sorted_counts)
plt.xticks(rotation=45, ha='right')
plt.title('Parts of Speech Frequency in Training Data (Sorted by Frequency)')
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{plot_dir}/pos_frequency.png")
plt.show()

# 4.3: Word Frequency Distribution Plot (excluding stopwords and punctuation)
word_freq = Counter(filtered_all_words)
common_words = word_freq.most_common(20)
words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title('Top 20 Most Common Words in Training Data (excluding punctuation, articles, conjunctions)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.savefig(f"{plot_dir}/word_frequency_filtered.png")
plt.show()

# 4.4: Word Frequency Distribution Plot (including everything)
all_words_including_everything = word_tokenize(all_text)
word_freq_including_everything = Counter(all_words_including_everything)
common_words_including_everything = word_freq_including_everything.most_common(20)

words_including_everything, counts_including_everything = zip(*common_words_including_everything)
plt.figure(figsize=(12, 6))
plt.bar(words_including_everything, counts_including_everything)
plt.xticks(rotation=45)
plt.title('Top 20 Most Common Words in Training Data (including everything)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.savefig(f"{plot_dir}/word_frequency_unfiltered.png")
plt.show()
