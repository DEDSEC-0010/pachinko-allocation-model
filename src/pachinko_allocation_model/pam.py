import numpy as np
from typing import List, Dict, Tuple, Optional
import scipy.special as special
from collections import Counter
import warnings
import logging


class PachinkoAllocationModel:
    def __init__(
        self,
        n_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 1.0,
        max_iterations: int = 100,
        random_state: Optional[int] = None,
        min_word_count: Optional[int] = None,
    ):
        """
        Enhanced Pachinko Allocation Model (PAM) for advanced topic modeling.

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
        random_state : int, optional (default=None)
            Random seed for reproducibility
        min_word_count : int, optional (default=None)
            Minimum word count for inclusion in vocabulary
        """
        # Input validation
        if n_topics <= 0:
            raise ValueError("Number of topics must be positive")
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Hyperparameters must be positive")

        # Set random seed
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        # Model parameters
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.min_word_count = min_word_count

        # Logging setup
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Model state
        self.topic_word_distribution = None
        self.document_topic_distribution = None
        self.topic_correlation_matrix = None
        self.vocab = None
        self.vocab_dict = None

    def _preprocess_documents(
        self, documents: List[List[str]]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Preprocess documents with robust vocabulary creation.

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

        # Determine minimum word count
        if self.min_word_count is None:
            min_count = max(1, len(documents) // 100)
        else:
            min_count = self.min_word_count

        # Filter vocabulary
        self.vocab = [word for word, count in word_counts.items() if count >= min_count]

        if not self.vocab:
            raise ValueError("No words meet the minimum count threshold")

        # Create vocabulary dictionary
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}
        self.logger.info(f"Vocabulary size: {len(self.vocab)}")

        # Create document-word matrix
        doc_word_matrix = np.zeros((len(documents), len(self.vocab)), dtype=int)
        for doc_idx, doc in enumerate(documents):
            for word in doc:
                if word in self.vocab_dict:
                    doc_word_matrix[doc_idx, self.vocab_dict[word]] += 1

        return doc_word_matrix, self.vocab_dict

    def fit(self, documents: List[List[str]]) -> "PachinkoAllocationModel":
        """
        Fit the Pachinko Allocation Model using enhanced Gibbs Sampling.

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
        try:
            doc_word_matrix, _ = self._preprocess_documents(documents)
        except ValueError as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

        num_docs, num_words = doc_word_matrix.shape
        self.logger.info(
            f"Training on {num_docs} documents with {num_words} unique words"
        )

        # Initialize topic assignments with improved randomization
        topic_assignments = np.random.randint(
            0, self.n_topics, size=(num_docs, num_words)
        )

        # Initialize count matrices with small epsilon for numerical stability
        epsilon = 1e-10
        doc_topic_counts = np.ones((num_docs, self.n_topics)) * (self.alpha + epsilon)
        topic_word_counts = np.ones((self.n_topics, num_words)) * (self.beta + epsilon)
        topic_counts = np.sum(topic_word_counts, axis=1)

        # Gibbs sampling with logging and potential early stopping
        for iteration in range(self.max_iterations):
            changes = 0

            for doc_idx in range(num_docs):
                for word_idx in range(num_words):
                    if doc_word_matrix[doc_idx, word_idx] > 0:
                        # Remove current topic assignment
                        current_topic = topic_assignments[doc_idx, word_idx]
                        doc_topic_counts[doc_idx, current_topic] -= 1
                        topic_word_counts[current_topic, word_idx] -= 1
                        topic_counts[current_topic] -= 1

                        # Calculate topic probabilities with log-space calculation
                        topic_probabilities = self._compute_topic_probabilities(
                            doc_topic_counts[doc_idx],
                            topic_word_counts[:, word_idx],
                            topic_counts,
                        )

                        # Sample new topic
                        new_topic = np.random.choice(
                            self.n_topics, p=topic_probabilities
                        )

                        # Update if topic changes
                        if new_topic != current_topic:
                            changes += 1

                        # Update assignments and counts
                        topic_assignments[doc_idx, word_idx] = new_topic
                        doc_topic_counts[doc_idx, new_topic] += 1
                        topic_word_counts[new_topic, word_idx] += 1
                        topic_counts[new_topic] += 1

            # Log iteration progress
            self.logger.info(
                f"Iteration {iteration + 1}/{self.max_iterations}: "
                f"Topic assignments changed: {changes}"
            )

            # Optional early stopping
            if changes < num_docs * num_words * 0.01:
                self.logger.info("Early stopping: Convergence reached")
                break

        # Compute final distributions with smoothing
        self.document_topic_distribution = (
            doc_topic_counts / doc_topic_counts.sum(axis=1)[:, np.newaxis]
        )
        self.topic_word_distribution = (
            topic_word_counts / topic_word_counts.sum(axis=1)[:, np.newaxis]
        )

        # Compute topic correlation with improved numerical stability
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

        # Compute log-space calculation with offsetting to prevent negative values
        log_probabilities = (
            np.log(doc_topic_counts + self.alpha)
            + np.log(topic_word_counts + self.beta)
            - np.log(topic_counts + len(self.vocab) * self.beta)
        )

        # Subtract the maximum log probability to prevent numerical instability
        log_probabilities -= log_probabilities.max()

        # Exponentiate to ensure non-negative values
        probabilities = np.exp(log_probabilities)

        # Add a small epsilon to prevent zero probabilities
        probabilities += 1e-10

        # Normalize probabilities
        return probabilities / probabilities.sum()

    def _compute_topic_correlation(
        self, document_topic_distribution: np.ndarray
    ) -> np.ndarray:
        """
        Compute topic correlation matrix with improved numerical handling.

        Parameters:
        -----------
        document_topic_distribution : np.ndarray
            Distribution of topics across documents

        Returns:
        --------
        correlation_matrix : np.ndarray
            Correlation matrix between topics
        """
        # Add small epsilon to prevent numerical instability
        epsilon = 1e-8
        correlation_matrix = np.corrcoef(document_topic_distribution.T + epsilon)
        return correlation_matrix

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transform new documents to topic distributions with robust inference.

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

        # Improved inference for new documents
        for doc_idx in range(num_docs):
            topic_probabilities = np.zeros(self.n_topics)

            for word_idx in range(num_words):
                if doc_word_matrix[doc_idx, word_idx] > 0:
                    # Compute word-topic probabilities
                    word_topic_probs = (
                        self.topic_word_distribution[:, word_idx]
                        * doc_word_matrix[doc_idx, word_idx]
                    )
                    topic_probabilities += word_topic_probs

            # Normalize with Dirichlet smoothing
            doc_topic_distributions[doc_idx] = (
                topic_probabilities / topic_probabilities.sum()
            )

        return doc_topic_distributions

    def get_top_words(self, n_words: int = 10) -> List[List[str]]:
        """
        Get top words for each topic with robust handling.

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

        # Ensure n_words doesn't exceed vocabulary size
        n_words = min(n_words, len(self.vocab))

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
        try:
            top_words = self.get_top_words(n_words)
            for idx, words in enumerate(top_words):
                print(f"Topic {idx + 1}: {', '.join(words)}")
        except Exception as e:
            self.logger.error(f"Error printing topics: {e}")


def pachinko_allocation_example():
    """
    Demonstration of Pachinko Allocation Model.

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
    pam = PachinkoAllocationModel(n_topics=3, random_state=42, max_iterations=150)
    pam.fit(documents)

    # Print topics
    print("Discovered Topics:")
    pam.print_topics()

    return "Pachinko Allocation Model - Topic Modeling Demonstration"


# Example usage
if __name__ == "__main__":
    pachinko_allocation_example()
