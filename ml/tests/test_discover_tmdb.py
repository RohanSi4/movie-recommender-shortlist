import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "discover_tmdb.py"
SPEC = importlib.util.spec_from_file_location("discover_tmdb", MODULE_PATH)
assert SPEC and SPEC.loader
discover_tmdb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(discover_tmdb)


class DiscoverTmdbTest(unittest.TestCase):
    def test_ml_genres_maps_tmdb_vocabulary(self) -> None:
        # 878 Science Fiction -> Sci-Fi, 53 Thriller -> Thriller, 12 Adventure.
        self.assertEqual(discover_tmdb.ml_genres([878, 53, 12]), "Sci-Fi|Thriller|Adventure")
        # 10751 Family -> Children, 10402 Music -> Musical.
        self.assertEqual(discover_tmdb.ml_genres([10751, 10402]), "Children|Musical")

    def test_ml_genres_drops_unmapped_and_defaults_when_empty(self) -> None:
        # 36 History and 10770 TV Movie have no MovieLens equivalent.
        self.assertEqual(discover_tmdb.ml_genres([36, 10770]), discover_tmdb.NO_GENRES)
        self.assertEqual(discover_tmdb.ml_genres([]), discover_tmdb.NO_GENRES)
        # A dropped genre next to a kept one keeps only the mapped one.
        self.assertEqual(discover_tmdb.ml_genres([36, 18]), "Drama")

    def test_ml_genres_dedupes(self) -> None:
        # 878 and 878 both map to Sci-Fi; the result must not repeat it.
        self.assertEqual(discover_tmdb.ml_genres([878, 878, 18]), "Sci-Fi|Drama")

    def test_build_row_shapes_a_catalog_record(self) -> None:
        result = {
            "id": 12345,
            "title": "A Recent Film",
            "genre_ids": [878, 28],
            "vote_average": 7.8,
            "vote_count": 1200,
            "popularity": 456.7,
            "poster_path": "/poster.jpg",
            "overview": "Something happens.",
            "release_date": "2025-03-14",
        }
        row = discover_tmdb.build_row(result, synthetic_base=100_000_000)
        self.assertEqual(row["movieId"], 100_012_345)
        self.assertEqual(row["tmdbId"], 12345)
        self.assertEqual(row["title"], "A Recent Film")
        self.assertEqual(row["genres"], "Sci-Fi|Action")
        self.assertEqual(row["tmdb_genres"], "Science Fiction|Action")
        self.assertEqual(row["tmdb_vote_avg"], 7.8)
        self.assertEqual(row["tmdb_release_date"], "2025-03-14")
        self.assertEqual(row["rating_count"], 0)

    def test_build_row_rejects_records_without_id_or_title(self) -> None:
        self.assertIsNone(discover_tmdb.build_row({"title": "No Id"}))
        self.assertIsNone(discover_tmdb.build_row({"id": 5, "genre_ids": [18]}))

    def test_synthetic_ids_stay_clear_of_movielens(self) -> None:
        # MovieLens ids top out near 292,757; synthetic ids must not collide.
        row = discover_tmdb.build_row({"id": 1, "title": "x"})
        self.assertGreater(row["movieId"], 300_000)


if __name__ == "__main__":
    unittest.main()
