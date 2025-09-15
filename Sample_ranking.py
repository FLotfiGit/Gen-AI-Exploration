from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example ticket issues
tickets = [
    "Cannot login to my account",
    "Login page not working",
    "Unable to reset password",
    "Password reset link is broken",
    "Error when uploading files",
    "File upload fails with error 500"
]

# Vectorize tickets
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(tickets)

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Categorize tickets by clustering similar ones (simple threshold-based)
def categorize_tickets(sim_matrix, threshold=0.5):
    categories = []
    assigned = set()
    for i in range(len(sim_matrix)):
        if i in assigned:
            continue
        group = [i]
        for j in range(i+1, len(sim_matrix)):
            if sim_matrix[i][j] > threshold:
                group.append(j)
                assigned.add(j)
        categories.append(group)
        assigned.add(i)
    return categories

categories = categorize_tickets(similarity_matrix)

# Print categorized tickets
for idx, group in enumerate(categories):
    print(f"Category {idx+1}:")
    for ticket_idx in group:
        print("  -", tickets[ticket_idx])

# Reranking within each category (by similarity to the first ticket in the group)
for idx, group in enumerate(categories):
    ref_idx = group[0]
    scores = [(tickets[i], similarity_matrix[ref_idx][i]) for i in group]
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nReranked Category {idx+1}:")
    for ticket, score in scores:
        print(f"  - {ticket} (score: {score:.2f})")

def get_ticket_categories(tickets, threshold=0.5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(tickets)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    categories = categorize_tickets(similarity_matrix, threshold)
    return [[tickets[i] for i in group] for group in categories]

def count_tickets(tickets):
    """Return the number of ticket issues."""
    return len(tickets)

def get_longest_ticket(tickets):
    """Return the ticket issue with the most characters."""
    return max(tickets, key=len) if tickets else None

def get_shortest_ticket(tickets):
    """Return the ticket issue with the fewest characters."""
    return min(tickets, key=len) if tickets else None

def filter_tickets_by_keyword(tickets, keyword):
    """Return a list of tickets containing the given keyword."""
    return [ticket for ticket in tickets if keyword.lower() in ticket.lower()]

def count_tickets_with_numbers(tickets):
    """Return the number of tickets containing at least one digit."""
    import re
    return sum(1 for ticket in tickets if re.search(r'\d', ticket))

def tickets_to_uppercase(tickets):
    """Return a list of tickets converted to uppercase."""
    return [ticket.upper() for ticket in tickets]

def reverse_tickets(tickets):
    """Return a list of tickets with their text reversed."""
    return [ticket[::-1] for ticket in tickets]

def get_tickets_with_word(tickets, word):
    """Return tickets containing the exact word (case-insensitive)."""
    return [ticket for ticket in tickets if word.lower() in ticket.lower().split()]

