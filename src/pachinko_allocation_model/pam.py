import numpy as np
from typing import List, Dict, Tuple
import scipy.special as special
from collections import Counter
import warnings


class PachinkoAllocationModel:
    def __init__(
        self,
        n_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 1.0,
        max_iterations: int = 100,
    ):
        """
        Pachinko Allocation Model (PAM) for advanced topic modeling.

        Parameters:
        -----------
        n_topics : int, optional (default=10)
            Number of topics in the model
        alpha : float, optional (default=0.1)
            Dirichlet distribution parameter for document-topic distribution
        beta : float, optional (default=0.01)
            Dirichlet distribution parameter for topic-word distribution
        gamma : float, optional (default=1.0)
            Hyperparameter for topic correlation
        max_iterations : int, optional (default=100)
            Maximum number of iterations for model training
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations

        # Model parameters to be learned
        self.topic_word_distribution = None
        self.document_topic_distribution = None
        self.topic_correlation_matrix = None

        # Vocabulary and document tracking
        self.vocab = None
        self.vocab_dict = None

    def _preprocess_documents(
        self, documents: List[List[str]]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Preprocess documents by creating a vocabulary and converting to numerical representation.

        Parameters:
        -----------
        documents : List[List[str]]
            List of tokenized documents

        Returns:
        --------
        doc_word_matrix : np.ndarray
            Numerical representation of documents
        vocab_dict : Dict[str, int]
            Mapping of words to their index
        """
        # Create vocabulary
        all_words = [word for doc in documents for word in doc]
        word_counts = Counter(all_words)

        # Filter out very rare words (optional)
        min_count = max(1, len(documents) // 100)
        self.vocab = [word for word, count in word_counts.items() if count >= min_count]
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}

        # Create document-word matrix
        doc_word_matrix = np.zeros((len(documents), len(self.vocab)), dtype=int)
        for doc_idx, doc in enumerate(documents):
            for word in doc:
                if word in self.vocab_dict:
                    doc_word_matrix[doc_idx, self.vocab_dict[word]] += 1

        return doc_word_matrix, self.vocab_dict

    def fit(self, documents: List[List[str]]) -> "PachinkoAllocationModel":
        """
        Fit the Pachinko Allocation Model to the documents using Gibbs Sampling.

        Parameters:
        -----------
        documents : List[List[str]]
            List of tokenized documents

        Returns:
        --------
        self : PachinkoAllocationModel
            Fitted model instance
        """
        # Preprocess documents
        doc_word_matrix, vocab_dict = self._preprocess_documents(documents)
        num_docs, num_words = doc_word_matrix.shape

        # Initialize topic assignments
        topic_assignments = np.random.randint(
            0, self.n_topics, size=(num_docs, len(vocab_dict))
        )

        # Initialize count matrices
        doc_topic_counts = np.zeros((num_docs, self.n_topics)) + self.alpha
        topic_word_counts = np.zeros((self.n_topics, num_words)) + self.beta
        topic_counts = np.zeros(self.n_topics) + num_words * self.beta

        # Gibbs sampling
        for _ in range(self.max_iterations):
            for doc_idx in range(num_docs):
                for word_idx in range(num_words):
                    if doc_word_matrix[doc_idx, word_idx] > 0:
                        # Remove current topic assignment
                        current_topic = topic_assignments[doc_idx, word_idx]
                        doc_topic_counts[doc_idx, current_topic] -= 1
                        topic_word_counts[current_topic, word_idx] -= 1
                        topic_counts[current_topic] -= 1

                        # Calculate topic probabilities
                        topic_probabilities = self._compute_topic_probabilities(
                            doc_topic_counts[doc_idx],
                            topic_word_counts[:, word_idx],
                            topic_counts,
                        )

                        # Sample new topic
                        new_topic = np.random.choice(
                            self.n_topics, p=topic_probabilities
                        )

                        # Update assignments and counts
                        topic_assignments[doc_idx, word_idx] = new_topic
                        doc_topic_counts[doc_idx, new_topic] += 1
                        topic_word_counts[new_topic, word_idx] += 1
                        topic_counts[new_topic] += 1

        # Compute final distributions
        self.document_topic_distribution = (
            doc_topic_counts / doc_topic_counts.sum(axis=1)[:, np.newaxis]
        )
        self.topic_word_distribution = (
            topic_word_counts / topic_word_counts.sum(axis=1)[:, np.newaxis]
        )

        # Compute topic correlation matrix
        self.topic_correlation_matrix = self._compute_topic_correlation(
            self.document_topic_distribution
        )

        return self

    def _compute_topic_probabilities(
        self,
        doc_topic_counts: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Compute topic probabilities for Gibbs sampling.

        Parameters:
        -----------
        doc_topic_counts : np.ndarray
            Count of topics in the current document
        topic_word_counts : np.ndarray
            Count of words in each topic
        topic_counts : np.ndarray
            Total count of words in each topic

        Returns:
        --------
        topic_probabilities : np.ndarray
            Normalized probabilities for each topic
        """
        # Compute probability proportional to joint probability
        probabilities = (
            (doc_topic_counts + self.alpha)
            * (topic_word_counts + self.beta)
            / (topic_counts + self.vocab.size * self.beta)
        )

        # Normalize probabilities
        return probabilities / probabilities.sum()

    def _compute_topic_correlation(
        self, document_topic_distribution: np.ndarray
    ) -> np.ndarray:
        """
        Compute topic correlation matrix based on document topic distributions.

        Parameters:
        -----------
        document_topic_distribution : np.ndarray
            Distribution of topics across documents

        Returns:
        --------
        correlation_matrix : np.ndarray
            Correlation matrix between topics
        """
        return np.corrcoef(document_topic_distribution.T)

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transform new documents to topic distributions.

        Parameters:
        -----------
        documents : List[List[str]]
            List of tokenized documents to transform

        Returns:
        --------
        topic_distributions : np.ndarray
            Topic distributions for input documents
        """
        if self.vocab is None:
            raise ValueError("Model must be fitted before transformation")

        # Preprocess documents
        doc_word_matrix, _ = self._preprocess_documents(documents)
        num_docs, num_words = doc_word_matrix.shape

        # Initialize topic distributions
        doc_topic_distributions = np.zeros((num_docs, self.n_topics))

        # Inference for new documents
        for doc_idx in range(num_docs):
            # Use existing topic-word distributions for inference
            topic_probabilities = np.zeros(self.n_topics)

            for word_idx in range(num_words):
                if doc_word_matrix[doc_idx, word_idx] > 0:
                    # Compute topic probabilities based on existing topic-word distributions
                    word_topic_probs = self.topic_word_distribution[:, word_idx] * (
                        doc_word_matrix[doc_idx, word_idx]
                    )
                    topic_probabilities += word_topic_probs

            # Normalize and smooth
            doc_topic_distributions[doc_idx] = (
                topic_probabilities / topic_probabilities.sum()
            )

        return doc_topic_distributions

    def get_top_words(self, n_words: int = 10) -> List[List[str]]:
        """
        Get top words for each topic.

        Parameters:
        -----------
        n_words : int, optional (default=10)
            Number of top words to retrieve for each topic

        Returns:
        --------
        top_words : List[List[str]]
            Top words for each topic
        """
        if self.topic_word_distribution is None:
            raise ValueError("Model must be fitted before retrieving top words")

        top_words = []
        for topic_idx in range(self.n_topics):
            # Get indices of top words for this topic
            top_word_indices = np.argsort(self.topic_word_distribution[topic_idx])[
                -n_words:
            ][::-1]

            # Map indices back to words
            topic_top_words = [self.vocab[idx] for idx in top_word_indices]
            top_words.append(topic_top_words)

        return top_words

    def print_topics(self, n_words: int = 10):
        """
        Print top words for each topic.

        Parameters:
        -----------
        n_words : int, optional (default=10)
            Number of top words to print for each topic
        """
        top_words = self.get_top_words(n_words)
        for idx, words in enumerate(top_words):
            print(f"Topic {idx + 1}: {', '.join(words)}")


def pachinko_allocation_example():
    """
    Example demonstration of Pachinko Allocation Model.

    Returns:
    --------
    str
        A simple demonstration message
    """
    # Sample documents
    documents = [
        ["machine", "learning", "algorithm", "data", "science"],
        ["neural", "network", "deep", "learning", "ai"],
        ["statistical", "analysis", "probability", "data", "science"],
        ["computer", "vision", "image", "processing", "machine", "learning"],
        ["natural", "language", "processing", "nlp", "ai", "deep", "learning"],
    ]

    # Create and fit PAM model
    pam = PachinkoAllocationModel(n_topics=3)
    pam.fit(documents)

    # Print topics
    print("Discovered Topics:")
    pam.print_topics()

    return "Pachinko Allocation Model - Topic Modeling Demonstration"


# Example usage
if __name__ == "__main__":
    pachinko_allocation_example()
