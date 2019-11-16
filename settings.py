import os


# data and path info
RAW_DATA_FILE = "amzn_clothing_reviews.txt"
PROC_DATA_FILE = "clothing_reviews.csv"
SEED = 1
SAMPLE_FRAC = .6

# model hyperparams
VECTORIZER_CUTOFF = 30
HIDDEN_DIM = 200
STOPWORDS = [ "ourselves", "hers", "between", "yourself", "but", "again",
                "there", "about", "once", "during", "out", "very", "having",
                "with", "they", "own", "an", "be", "some", "for", "do", "its",
                "yours", "such", "into", "of", "most", "itself", "other", "off",
                "is", "s", "am", "or", "who", "as", "from", "him", "each", "the",
                "themselves", "until", "below", "are", "we", "these", "your", "his",
                "through", "don", "nor", "me", "were", "her", "more", "himself",
                "this", "down", "should", "our", "their", "while", "above", "both",
                "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any",
                "before", "them", "same", "and", "been", "have", "in", "will", "on",
                "does", "yourselves", "then", "that", "because", "what", "over", "why",
                "so", "can", "did", "not", "now", "under", "he", "you", "herself",
                "has", "just", "where", "too", "only", "myself", "which", "those", "d",
                "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against",
                "a", "by", "doing", "it", "how", "further", "was", "here", "than", "ve" ]

# training hyperparams
BATCH_SIZE = 1024
EARLY_STOPPING_CRITERIA = 5
LEARNING_RATE = 0.001
NUM_EPOCHS = 8
DROP_LAST = True
DROPOUT_P = 0.7
THRESHOLD = 0.0