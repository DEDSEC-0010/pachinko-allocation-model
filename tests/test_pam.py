import unittest
import numpy as np
from pachinko_allocation_model import (
    PachinkoAllocationModel,
    pachinko_allocation_example,
)


class TestPAM(unittest.TestCase):
    def test_initialization(self):
        pam = PachinkoAllocationModel(n_topics=5)
        self.assertEqual(pam.n_topics, 5)

    def test_example_function(self):
        result = pachinko_allocation_example()
        self.assertIsInstance(result, str)

    def test_fit_method(self):
        # Sample documents
        documents = [
            ["machine", "learning", "algorithm"],
            ["data", "science", "python"],
            ["neural", "network", "deep", "learning"],
        ]

        pam = PachinkoAllocationModel(n_topics=3)
        pam.fit(documents)

        # Check if distribution is generated
        self.assertIsNotNone(pam.document_topic_distribution)
        self.assertIsNotNone(pam.topic_word_distribution)

    def test_transform_method(self):
        documents = [
            ["machine", "learning", "algorithm"],
            ["data", "science", "python"],
        ]

        pam = PachinkoAllocationModel(n_topics=3)
        pam.fit(documents)

        topic_distributions = pam.transform(documents)

        self.assertEqual(topic_distributions.shape[0], len(documents))
        self.assertEqual(topic_distributions.shape[1], pam.n_topics)

        # Check probability distribution sums to 1
        np.testing.assert_almost_equal(
            topic_distributions.sum(axis=1), np.ones(len(documents)), decimal=7
        )


if __name__ == "__main__":
    unittest.main()
