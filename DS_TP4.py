from functools import reduce
from collections import defaultdict

# Sample input data (list of sentences)
input_data = [
    "MapReduce is a programming model and a great tool",
    "It is used for processing and generating large datasets",
    "The MapReduce model involves two main steps: Map and Reduce"
]

# Step 1: Map - Extract words from each sentence
def mapper(sentence):
    words = sentence.split()
    word_counts = defaultdict(int)
    for word in words:
        # Use lowercase to count words case-insensitively
        word_counts[word.lower()] += 1
    return word_counts

# Step 2: Reduce - Sum up the counts for each word
def reducer(word_counts, other_word_counts):
    for word, count in other_word_counts.items():
        word_counts[word] += count
    return word_counts

# Map step
mapped_data = list(map(mapper, input_data))

# Flatten the list of dictionaries
flattened_data = reduce(reducer, mapped_data, defaultdict(int))

# Reduce step
total_word_count = sum(flattened_data.values())


# Calculate average word length
total_word_length = sum(len(word) * count for word, count in flattened_data.items())
average_word_length = total_word_length / total_word_count if total_word_count > 0 else 0

# Print the word count result and average word length
print("Word Count Result:")
for word, count in flattened_data.items():
    print(f"{word}: {count}")

print("\nAverage Word Length:", average_word_length)
