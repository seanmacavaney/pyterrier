import numpy as np
import json
import shutil
import sys
import pandas as pd
from pathlib import Path
import tempfile
import os
import more_itertools
import threading
from warnings import warn
from typing import Optional, Union, List
from enum import Enum
from collections import Counter
import pyterrier as pt
from pyterrier.datasets import Dataset
import functools


class TerrierStemmer(Enum):
    """
    Stemmer
    """
    none = 'none'
    danish_snowball = 'danish_snowball'
    dutch_snowball = 'dutch_snowball'
    english_snowball = 'english_snowball'
    finnish_snowball = 'finnish_snowball'
    french_snowball = 'french_snowball'
    german_snowball = 'german_snowball'
    hungarian_snowball = 'hungarian_snowball'
    italian_snowball = 'italian_snowball'
    norwegian_snowball = 'norwegian_snowball'
    porter = 'porter'
    portuguese_snowball = 'portuguese_snowball'
    romanian_snowball = 'romanian_snowball'
    russian_snowball = 'russian_snowball'
    spanish_snowball = 'spanish_snowball'
    swedish_snowball = 'swedish_snowball'
    tr_v2_porter = 'tr_v2_porter'
    tr_v2_weak_porter = 'tr_v2_weak_porter'
    turkish_snowball = 'turkish_snowball'
    weak_porter = 'weak_porter'

_STEMMER_MAP = {
    TerrierStemmer.none: '',
    TerrierStemmer.danish_snowball: 'DanishSnowballStemmer',
    TerrierStemmer.dutch_snowball: 'DutchSnowballStemmer',
    TerrierStemmer.english_snowball: 'EnglishSnowballStemmer',
    TerrierStemmer.finnish_snowball: 'FinnishSnowballStemmer',
    TerrierStemmer.french_snowball: 'FrenchSnowballStemmer',
    TerrierStemmer.german_snowball: 'GermanSnowballStemmer',
    TerrierStemmer.hungarian_snowball: 'HungarianSnowballStemmer',
    TerrierStemmer.italian_snowball: 'ItalianSnowballStemmer',
    TerrierStemmer.norwegian_snowball: 'NorwegianSnowballStemmer',
    TerrierStemmer.porter: 'PorterStemmer',
    TerrierStemmer.portuguese_snowball: 'PortugueseSnowballStemmer',
    TerrierStemmer.romanian_snowball: 'RomanianSnowballStemmer',
    TerrierStemmer.russian_snowball: 'RussianSnowballStemmer',
    TerrierStemmer.spanish_snowball: 'SpanishSnowballStemmer',
    TerrierStemmer.swedish_snowball: 'SwedishSnowballStemmer',
    TerrierStemmer.tr_v2_porter: 'TRv2PorterStemmer',
    TerrierStemmer.tr_v2_weak_porter: 'TRv2WeakPorterStemmer',
    TerrierStemmer.turkish_snowball: 'TurkishSnowballStemmer',
    TerrierStemmer.weak_porter: 'WeakPorterStemmer',
}
_STEMMER_MAP_INV = {v: k for k, v in _STEMMER_MAP.items()}




class TerrierStopwords(Enum):
    """
    Stemmer
    """
    terrier = 'terrier'
    none = 'none'


TERRIER_INDEX_DEFAULTS = {
    'stemmer': TerrierStemmer.porter,
    'stops': TerrierStopwords.terrier,
}


class TerrierIndex(pt.transformer.IterDictIndexerBase):
    def __init__(self,
                 path: str,
                 text_field: str = None,
                 stemmer: Optional[Union[TerrierStemmer, str]] = None,
                 stops: Optional[Union[TerrierStopwords, str]] = None,
                 threads: int = 1,
                 overwrite=False):
        self.path = path
        ppath = Path(path)
        if not ppath.isabs(): # If a relative path doesn't start with './', it will look under ./var/
            self.path = f'./{path}'
        if stemmer is not None: stemmer = TerrierStemmer(stemmer)
        if stops is not None: stops = TerrierStopwords(stops)
        if (ppath/'data.properties').exists():
            props = dict(l.rstrip('\n').split('=', 1) for l in (ppath/'data.properties').open() if '=' in l)
            termpipelines = tuple(props.get('termpipelines', '').split(','))
            stemmer_from_props = None
            stops_from_props = None
            if termpipelines == ('',):
                stemmer_from_props = TerrierStemmer.none
                stops_from_props = TerrierStopwords.none
            elif termpipelines == ('Stopwords',):
                stops_from_props = TerrierStopwords.terrier
                stemmer_from_props = TerrierStemmer.none
            elif len(termpipelines) == 1 and termpipelines[0] in _STEMMER_MAP_INV:
                stops_from_props = TerrierStopwords.none
                stemmer_from_props = _STEMMER_MAP_INV[termpipelines[0]]
            elif len(termpipelines) == 2 and termpipelines[0] == 'Stopwords' and termpipelines[1] in _STEMMER_MAP_INV:
                stops_from_props = TerrierStopwords.terrier
                stemmer_from_props = _STEMMER_MAP_INV[termpipelines[1]]
            else:
                raise RuntimeError(f'Unsupported termpipelines={termpipelines}; you cannot use this index with the TerrierIndex() class. Consider using TerrierRetrieve directly.')
            if stemmer is None:
                stemmer = stemmer_from_props
            if stemmer != stemmer_from_props:
                warn(f'requested stemmer={stemmer}, but index was constructed with {stemmer_from_props}')
            if stops is None:
                stops = stops_from_props
            if stops != stops_from_props:
                warn(f'requested stemmer={stops}, but index was constructed with {stops_from_props}')
        if stemmer is None: stemmer = TERRIER_INDEX_DEFAULTS['stemmer']
        if stops is None: stops = TERRIER_INDEX_DEFAULTS['stops']
        self.text_field = text_field
        self.stemmer = stemmer
        self.threads = threads
        self.overwrite = overwrite
        self.stops = stops
        self._indexobj = None
        self._indexref = None

    def transform(self, *args, **kwargs):
        raise RuntimeError(f'You cannot use {self} itself as a transformer. Did you mean to call a ranking function like .bm25()?')

    def built(self) -> bool:
        return (Path(self.path)/'data.properties').exists()

    def index_ref(self):
        if not self._indexref:
            if not self.built():
                raise RuntimeError(f'Index {self} not yet built.')
            self._indexref = pt.IndexRef.of(str(self.path))
        return self._indexref

    def index_obj(self):
        if not self._indexobj:
            self._indexobj = pt.IndexFactory.of(self.index_ref())
        return self._indexobj

    def index(self, inp):
        indexer = pt.IterDictIndexer(str(self.path))
        return indexer.index(inp)

    def _get_termpipelines(self):
        pipeline = []
        if self.stops == TerrierStopwords.terrier:
            pipeline.append('Stopwords')
        if self.stemmer != TerrierStemmer.none:
            pipeline.append(_STEMMER_MAP[self.stemmer])
        return ','.join(pipeline)

    def bm25(self, k1=1.2, k3=8, b=0.75, num_results=1000, verbose=False, threads=None) -> pt.Transformer:
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='BM25',
            controls={'bm25.k_1': k1, 'bm25.k_3': k3, 'c': b})
        retr.verbose = verbose
        return retr

    def dph(self, num_results=1000, verbose=False, threads=None) -> pt.Transformer:
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='DPH')
        retr.verbose = verbose
        return retr

    def tf(self, num_results=1000, verbose=False, threads=None) -> pt.Transformer:
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='TF')
        retr.verbose = verbose
        return retr

    def tf_idf(self, k1=1.2, b=0.75, num_results=1000, verbose=False, threads=None) -> pt.Transformer:
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='TF_IDF',
            controls={'tf_idf.k_1': k1, 'tf_idf.b': b})
        retr.verbose = verbose
        return retr

    def pl2(self, c=1.0, num_results=1000, verbose=False, threads=None) -> pt.Transformer:
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='PL2',
            controls={'c': c})
        retr.verbose = verbose
        return retr

    def lmd(self, mu=2500., num_results=1000, verbose=False, threads=None):
        retr = pt.TerrierRetrieve(self.index_ref(), num_results=num_results, threads=threads or self.threads,
            properties={'termpipelines': self._get_termpipelines()},
            wmodel='DirichletLM',
            controls={'dirichletlm.mu': mu})
        retr.verbose = verbose
        return retr

    def num_terms(self) -> Optional[int]:
        if self.built():
            return self.index_obj().getCollectionStatistics().getNumberOfUniqueTerms()

    def num_docs(self) -> Optional[int]:
        if self.built():
            return self.index_obj().getCollectionStatistics().getNumberOfDocuments()

    def __repr__(self):
        return f'TerrierIndex({repr(self.path)})'

    @staticmethod
    def from_dataset(dataset: Union[str, Dataset], variant, version: str = 'latest', **kwargs):
        from pyterrier.batchretrieve import _from_dataset
        return _from_dataset(dataset, variant=variant, version=version, clz=TerrierIndex, **kwargs)
