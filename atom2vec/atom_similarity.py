"""
Main class for generating, querying atom vectors
"""

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, AnyStr

import numpy as np
from pymatgen.core import  Structure, Composition
from pymatgen.core.periodic_table import get_el_sp, Element, Species
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd


class AtomSimilarity:
    """
    Compute the similarity of atoms
    """

    def __init__(self, max_elements, atoms_vector, atoms_similarity):
        self._max_elements: int = max_elements
        self._k_dim = len(next(iter(atoms_vector.values())))
        self._atoms_vector: Dict[str, List[float]] = atoms_vector
        self._atoms_similarity: Dict[str, Dict[str, float]] = atoms_similarity

    @property
    def max_elements(self):
        return self._max_elements

    @property
    def k_dim(self):
        return self._k_dim

    @classmethod
    def _kwargs_from_structures(cls, structures: List[Structure], k_dim: int,
                                max_elements: int) -> Dict[str, Any]:
        compositions = [s.composition.reduced_composition for s in structures
                        if 1 < len(s.composition.elements) <= max_elements]

        env_dict = defaultdict(list)
        elements_set: Set[Element] = set()

        for composition in tqdm(compositions, desc="Generating the environment matrix"):
            composition_items = composition.items()
            for e, amount in composition_items:
                env_composition = composition - Composition({e: amount})
                env_dict[env_composition].append(e)
                elements_set.add(e)

        elements = {e: i for i, e in enumerate(list(elements_set))}

        env_matrix = np.zeros((len(elements), len(env_dict)), dtype=np.int8)
        for j, elements_ in enumerate(env_dict.values()):
            for e in elements_:
                i = elements[e]
                env_matrix[i][j] = 1

        u, d, v = svds(env_matrix.astype(dtype=np.float32), k=k_dim, which="LM")
        atoms_vector = u @ np.diag(d)
        atoms_vector_dict = {e.name: atoms_vector[i].tolist() for e, i in elements.items()}

        atoms_similarity = cosine_similarity(atoms_vector)
        atoms_similarity_dict = {e.name: {_e.name: float(atoms_similarity[i, _i])
                                          for _e, _i in elements.items()}
                                 for e, i in elements.items()}

        return {
            "atoms_similarity": atoms_similarity_dict,
            "atoms_vector": atoms_vector_dict,
            "max_elements": max_elements,
        }

    @classmethod
    def from_structures(cls, structures: List[Structure],
                        k_dim: int, max_elements: int):
        """
        Generating atom vectors and atom similarity matrix from list of :obj:`pymatgen.core.Structure`

        Args:
            structures: list of structures
            k_dim: the dimension of atom vectors, recommended value: [50, 100, 300]
            max_elements: if the number of elements in a structure exceeds `max_elements`, it will be
                automatically ignored to save space as a atomic environment of too many elements are
                very rare, recommended value: 3
        """
        return cls(**cls._kwargs_from_structures(structures=structures,
                                                 k_dim=k_dim, max_elements=max_elements))

    def get_atom_vector(self, item: Union[Element, AnyStr, int]) -> List[float]:
        """
        Query atom vector

        Raises:
            KeyError: when the element requested do not exist.
        """
        e = get_el_sp(item)
        try:
            return self._atoms_vector[e.name]
        except KeyError as err:
            err.args = ("Not such element as {}".format(item),)
            raise

    def get_atom_similarity(self, item: Tuple[Union[Element, AnyStr, int]]) -> float:
        """
        Query cosine similarity of two elements

        The data type of element can be:
            1) `pymatgen.core.Element` enum type
            2) element string
            3) atom index in the periodic table

        If neither elements are not presented in the training set, -1. will be returned.
        """
        if not isinstance(item, tuple) and len(item) != 2:
            raise ValueError("Expect two element tuple, but get {}".format(item))
        e_1, e_2 = get_el_sp(item[0]), get_el_sp(item[1])
        return self._atoms_similarity.get(e_1.name, {}).get(e_2.name, -1.)

    def __getitem__(self, item: Tuple[Union[Element, AnyStr, int]]):
        return self.get_atom_similarity(item)

    def as_dict(self):
        return {
            "atoms_similarity": self._atoms_similarity,
            "atoms_vector": self._atoms_vector,
            "max_elements": self._max_elements,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_csv_vectors(self, filename:str):
        """ Save atom vectors to csv file """
        sorted_atoms_dict =dict(sorted(self._atoms_vector.items(), key=lambda v:Element(v[0]).number ))

        df = pd.DataFrame.from_dict(sorted_atoms_dict, orient="index")
        df.to_csv(filename, index_label="element")

        


class SpeciesSimilarity:
    """
    Compute the similarity of species
    """

    def __init__(self, max_species, species_vector, species_similarity):
        self._max_species: int = max_species
        self._k_dim = len(next(iter(species_vector.values())))
        self._species_vector: Dict[str, List[float]] = species_vector
        self._species_similarity: Dict[str, Dict[str, float]] = species_similarity

    @property
    def max_species(self):
        return self._max_species

    @property
    def k_dim(self):
        return self._k_dim

    @classmethod
    def _kwargs_from_structures(cls, structures: List[Structure], k_dim: int,
                                max_species: int) -> Dict[str, Any]:
        compositions = [s.composition.reduced_composition for s in structures
                        if 1 < len(s.composition.elements) <= max_species]

        env_dict = defaultdict(list)
        species_set: Set[Species] = set()

        for composition in tqdm(compositions, desc="Generating the environment matrix"):
            composition_items = composition.items()
            for s, amount in composition_items:
                env_composition = composition - Composition({s: amount})
                env_dict[env_composition].append(s)
                species_set.add(s)

        species = {s: i for i, s in enumerate(list(species_set))}

        env_matrix = np.zeros((len(species), len(env_dict)), dtype=np.int8)
        for j, species_ in enumerate(env_dict.values()):
            for s in species_:
                i = species[s]
                env_matrix[i][j] = 1

        u, d, v = svds(env_matrix.astype(dtype=np.float32), k=k_dim, which="LM")
        species_vector = u @ np.diag(d)
        species_vector_dict = {s.to_pretty_string(): species_vector[i].tolist() for s, i in species.items()}

        species_similarity = cosine_similarity(species_vector)
        species_similarity_dict = {s.to_pretty_string(): {_s.to_pretty_string(): float(species_similarity[i, _i])
                                          for _s, _i in species.items()}
                                 for s, i in species.items()}

        return {
            "species_similarity": species_similarity_dict,
            "species_vector": species_vector_dict,
            "max_species": max_species,
        }

    @classmethod
    def from_structures(cls, structures: List[Structure],
                        k_dim: int, max_species: int):
        """
        Generating species vectors and species similarity matrix from list of :obj:`pymatgen.core.Structure`

        Args:
            structures: list of structures
            k_dim: the dimension of species vectors, recommended value: [50, 100, 300]
            max_elements: if the number of species in a structure exceeds `max_species`, it will be
                automatically ignored to save space as a species environment of too many species are
                very rare, recommended value: 3
        """
        return cls(**cls._kwargs_from_structures(structures=structures,
                                                 k_dim=k_dim, max_species=max_species))

    def get_species_vector(self, item: Union[Species, AnyStr, int]) -> List[float]:
        """
        Query species vector

        Raises:
            KeyError: when the species requested do not exist.
        """
        e = get_el_sp(item)
        try:
            return self._species_vector[e.to_pretty_string()]
        except KeyError as err:
            err.args = ("Not such species as {}".format(item),)
            raise

    def get_species_similarity(self, item: Tuple[Union[Species, AnyStr]]) -> float:
        """
        Query cosine similarity of two species

        The data type of species can be:
            1) `pymatgen.core.Species` enum type
            2) species string

        If neither species are not presented in the training set, -1. will be returned.
        """
        if not isinstance(item, tuple) and len(item) != 2:
            raise ValueError(f"Expect two species tuple, but get {item}")
        e_1, e_2 = get_el_sp(item[0]), get_el_sp(item[1])
        return self._species_similarity.get(e_1.to_pretty_string(), {}).get(e_2.to_pretty_string(), -1.)

    def __getitem__(self, item: Tuple[Union[Species, AnyStr]]):
        return self.get_species_similarity(item)

    def as_dict(self):
        return {
            "species_similarity": self._species_similarity,
            "species_vector": self._species_vector,
            "max_species": self._max_species,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_csv_vectors(self, filename: str):
        """
        Save species vectors to csv file
        """
        sorted_species_dict =dict(sorted(self._species_vector.items(), key=lambda v:Species.from_string(v[0]).number ))

        df = pd.DataFrame.from_dict(sorted_species_dict, orient="index")
        df.to_csv(filename, index_label="species")
