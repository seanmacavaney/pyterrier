from .base import BaseTestCase
import pandas as pd
import pyterrier as pt

class TestTransformers(BaseTestCase):

    def test_id(self):
        dataset = pt.datasets.get_dataset("vaswani")
        dph = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH", id="dph")
        pipe = dph % 10
        self.assertIsNotNone(dph.get_transformer("dph"))
        self.assertIsNotNone(pipe.get_transformer("dph"))
        self.assertIsNone(dph.get_transformer("bla"))
        self.assertIsNone(pipe.get_transformer("bla"))

    def test_params(self):
        dataset = pt.datasets.get_dataset("vaswani")
        bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25", id="bm25", controls={"c" : 0.1}, properties={"bm25.k_1" : 1.2})
        self.assertEqual(0.1, bm25.get_parameter("c"))
        self.assertEqual("BM25", bm25.get_parameter("wmodel"))
        self.assertEqual(1.2, bm25.get_parameter("bm25.k_1"))

        bm25.set_parameter("c", 0.2)
        bm25.set_parameter("wmodel", "BM25F")
        bm25.set_parameter("bm25.k_1", 1.3)

        self.assertEqual(0.2, bm25.get_parameter("c"))
        self.assertEqual("BM25F", bm25.get_parameter("wmodel"))
        self.assertEqual(1.3, bm25.get_parameter("bm25.k_1"))

        with self.assertRaises(ValueError):
            bm25.get_parameter("d")

        with self.assertRaises(ValueError):
            bm25.set_parameter("d", 1.3)
