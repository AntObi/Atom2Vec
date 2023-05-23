import json
import os
import unittest
from io import StringIO
from itertools import product

from pymatgen.core import Element, Species
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.bond_valence import BVAnalyzer
from atom2vec import AtomSimilarity, SpeciesSimilarity


class TestAtomSimilarityQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.k_dim = 20
        cls.max_element = 3
        API_KEY = os.environ["MP_API_KEY"]

        if len(API_KEY)==16:

            with MPRester(os.environ["MP_API_KEY"]) as mpr:
                criteria = {
                    "nelements": 2,
                    "e_above_hull": {"$lte": 0.}
                }
                properties = [
                    "structure",
                ]
                entries = mpr.query(criteria=criteria, properties=properties, mp_decode=True)

            structures = [entry["structure"] for entry in entries]
        elif len(API_KEY)==32:
            with MPRester(API_KEY) as mpr:
                docs = mpr.summary.search(
                    num_elements=2, 
                    energy_above_hull=0, 
                    fields=["structure"])
            structures = [doc.structure for doc in docs]
        cls.atom_similarity = AtomSimilarity.from_structures(structures=structures,
                                                             k_dim=cls.k_dim,
                                                             max_elements=cls.max_element)

    def test_k_dim(self):
        self.assertEqual(self.k_dim, self.atom_similarity.k_dim)

    def test_same_atom_similarity(self):
        elements = self.atom_similarity._atoms_vector.keys()
        for element in elements:
            similarity = self.atom_similarity[element, element]
            self.assertAlmostEqual(1., similarity,
                                   msg="Similarity of same element is not 1.", delta=1e-5)

    def test_atom_similarity(self):
        elements = ("Fe", 27, Element("Ni"))

        query_tuples = product(elements, elements)
        for q in query_tuples:
            self.assertTrue(-1 - 1e-5 <= self.atom_similarity.__getitem__(q) <= 1. + 1e-5,
                            msg="Similarity of element {0}, {1} not in -1 ~ 1.".format(*q))

    def test_not_exist_atom_similarity(self):
        elements = (118, "Na")

        self.assertEqual(-1, self.atom_similarity.__getitem__(elements),
                         msg="Similarity for non-existing element should be -1.")

    def test_atom_vector(self):
        elements = ("Na", 12, Element("Al"))

        for element in elements:
            self.assertEqual(self.k_dim, len(self.atom_similarity.get_atom_vector(element)),
                             msg="Wrong size for queried vector")

    def test_not_exist_atom_vector(self):
        self.assertRaises(KeyError, self.atom_similarity.get_atom_vector, 118)

    def test_load_dump(self):
        # dump to json format
        atom_similarity_dict = self.atom_similarity.as_dict()
        atom_similarity_json = StringIO()
        json.dump(atom_similarity_dict, atom_similarity_json)

        atom_similarity_json.seek(0)

        # load json string to new AtomSimilarity object
        new_atom_similarity_dict = json.load(atom_similarity_json)
        new_atom_similarity = AtomSimilarity.from_dict(new_atom_similarity_dict)

        self.assertEqual(new_atom_similarity._atoms_vector, self.atom_similarity._atoms_vector,
                         msg="Unmatched atom vector after dumping and loading.")
        self.assertEqual(new_atom_similarity._atoms_similarity, self.atom_similarity._atoms_similarity,
                         msg="Unmatched atom similarity after dumping and loading.")

        self.assertEqual(new_atom_similarity.k_dim, self.atom_similarity.k_dim,
                         msg="Unmatched k_dim after dumping and loading.")
        self.assertEqual(new_atom_similarity.max_elements, self.atom_similarity.max_elements,
                         msg="Unmatched max_elements after dumping and loading.")

class TestSpeciesSimilarityQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.k_dim = 20
        cls.max_species = 3
        API_KEY = os.environ["MP_API_KEY"]

        if len(API_KEY)==16:

            with MPRester(os.environ["MP_API_KEY"]) as mpr:
                criteria = {
                    "nelements": 2,
                    "e_above_hull": {"$lte": 0.},
                    "band_gap": {"$gt": 2.}
                }
                properties = [
                    "structure",
                ]
                entries = mpr.query(criteria=criteria, properties=properties, mp_decode=True)

            structures = [entry["structure"] for entry in entries]
        elif len(API_KEY)==32:
            with MPRester(API_KEY) as mpr:
                docs = mpr.summary.search(
                    num_elements=2, 
                    energy_above_hull=0,
                    band_gap=(2,None) ,
                    fields=["structure"])
            structures = [doc.structure for doc in docs]
        
        def decorate_structure(structure):
            bv = BVAnalyzer()
            try:
                return bv.get_oxi_state_decorated_structure(structure)
            except ValueError:
                return None
        structures = [decorate_structure(structure) for structure in structures]
        structures = [structure for structure in structures if structure is not None]
        cls.species_similarity = SpeciesSimilarity.from_structures(structures=structures,
                                                             k_dim=cls.k_dim,
                                                             max_species=cls.max_species)

    def test_k_dim(self):
        self.assertEqual(self.k_dim, self.species_similarity.k_dim)

    def test_same_species_similarity(self):
        species = self.species_similarity._species_vector.keys()
        for specie in species:
            similarity = self.species_similarity[specie, specie]
            self.assertAlmostEqual(1., similarity,
                                   msg="Similarity of same specie is not 1.", delta=1e-5)

    def test_species_similarity(self):
        species = ("Fe2+", Species.from_string("Co2+"), Species("Ni",2))

        query_tuples = product(species, species)
        for q in query_tuples:
            self.assertTrue(-1 - 1e-5 <= self.species_similarity.__getitem__(q) <= 1. + 1e-5,
                            msg="Similarity of element {0}, {1} not in -1 ~ 1.".format(*q))

    def test_not_exist_species_similarity(self):
        species = ("H7+", "Na7-")

        self.assertEqual(-1, self.species_similarity.__getitem__(species),
                         msg="Similarity for non-existing element should be -1.")

    def test_species_vector(self):
        species = ("Na+", "Mg2+", Species.from_string("Al3+"))

        for specie in species:
            self.assertEqual(self.k_dim, len(self.species_similarity.get_species_vector(specie)),
                             msg="Wrong size for queried vector")

    def test_not_exist_species_vector(self):
        self.assertRaises(KeyError, self.species_similarity.get_species_vector, "Na7+")

    def test_load_dump(self):
        # dump to json format
        species_similarity_dict = self.species_similarity.as_dict()
        species_similarity_json = StringIO()
        json.dump(species_similarity_dict, species_similarity_json)

        species_similarity_json.seek(0)

        # load json string to new AtomSimilarity object
        new_species_similarity_dict = json.load(species_similarity_json)
        new_species_similarity = SpeciesSimilarity.from_dict(new_species_similarity_dict)

        self.assertEqual(new_species_similarity._species_vector, self.species_similarity._species_vector,
                         msg="Unmatched species vector after dumping and loading.")
        self.assertEqual(new_species_similarity._species_similarity, self.species_similarity._species_similarity,
                         msg="Unmatched species similarity after dumping and loading.")

        self.assertEqual(new_species_similarity.k_dim, self.species_similarity.k_dim,
                         msg="Unmatched k_dim after dumping and loading.")
        self.assertEqual(new_species_similarity.max_species, self.species_similarity.max_species,
                         msg="Unmatched max_species after dumping and loading.")
