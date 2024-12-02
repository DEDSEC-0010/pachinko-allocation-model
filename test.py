from pachinko_allocation_model import PachinkoAllocationModel

# Sample documents
documents = [
    ["machine", "learning", "algorithm", "data", "science"],
    ["neural", "network", "deep", "learning", "ai"],
    # ... more documents
]

# Create and fit model
pam = PachinkoAllocationModel(n_topics=3)
pam.fit(documents)

# Print discovered topics
pam.print_topics()

# Transform new documents
new_docs = [["data", "analysis", "statistics"], ["neural", "network", "prediction"]]
topic_distributions = pam.transform(new_docs)
