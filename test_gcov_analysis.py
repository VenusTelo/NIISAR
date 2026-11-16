import base64
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
from flask_caching.backends.filesystemcache import FileSystemCache
from pyproj import Transformer
from shapely.geometry import box, mapping

import app
import gcov_analysis


def observation(date, path=4, frame=9, direction="ASCENDING"):
    return SimpleNamespace(geometry=mapping(box(-1, -1, 1, 1)), properties={
        "startTime": date + "T00:00:00Z",
        "pathNumber": path,
        "frameNumber": frame,
        "flightDirection": direction,
        "url": f"https://nisar.asf.earthdatacloud.nasa.gov/{date}.h5",
    })


def hdf_bytes(value, x_offset=0):
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as hdf:
        group = hdf.create_group(gcov_analysis.GCOV_GROUP)
        group["projection"] = 4326
        group["xCoordinates"] = np.arange(8) * 0.01 - 0.035 + x_offset
        group["yCoordinates"] = 0.035 - np.arange(8) * 0.01
        group["xCoordinateSpacing"] = 0.01
        group["yCoordinateSpacing"] = -0.01
        group["HVHV"] = np.full((8, 8), value, dtype="float32")
    return buffer.getvalue()


def projected_hdf_bytes(value):
    buffer = io.BytesIO()
    center_x, center_y = Transformer.from_crs(4326, 6933, always_xy=True).transform(-84.7, 15.6)
    with h5py.File(buffer, "w") as hdf:
        group = hdf.create_group(gcov_analysis.GCOV_GROUP)
        group["projection"] = 6933
        group["xCoordinates"] = center_x + (np.arange(16) - 7.5) * 30
        group["yCoordinates"] = center_y - (np.arange(16) - 7.5) * 30
        group["xCoordinateSpacing"] = 30
        group["yCoordinateSpacing"] = -30
        group["HVHV"] = np.full((16, 16), value, dtype="float32")
    return buffer.getvalue()


class GcovAnalysisTests(unittest.TestCase):
    def test_rejects_untrusted_url(self):
        with self.assertRaises(gcov_analysis.AnalysisError):
            gcov_analysis.h5_url({"url": "https://example.com/data.h5"})

    def test_uses_cmr_metadata_when_properties_are_missing(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]
        ])
        results = []
        for day in ("2025-01-01", "2025-01-03"):
            item = observation(day)
            item.properties["pathNumber"] = None
            item.properties["frameNumber"] = None
            item.umm = {"AdditionalAttributes": [
                {"Name": "TRACK_NUMBER", "Values": ["4"]},
                {"Name": "FRAME_NUMBER", "Values": ["9"]},
                {"Name": "FREQUENCY_A_POLARIZATION", "Values": ["HH", "HV"]},
            ]}
            results.append(item)
        with patch.object(gcov_analysis.asf, "search", return_value=results) as search:
            groups = gcov_analysis.fetch_area_observations(area)
        self.assertEqual(list(groups), [(4, 9, "ASCENDING")])
        self.assertEqual(sorted(groups[(4, 9, "ASCENDING")]), ["2025-01-01", "2025-01-03"])
        self.assertEqual(search.call_args.kwargs["shortName"], "NISAR_L2_GCOV_V1")

    def test_requires_hv_when_polarization_metadata_is_available(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]
        ])
        first = observation("2025-01-01")
        second = observation("2025-01-03")
        second.umm = {"AdditionalAttributes": [
            {"Name": "FREQUENCY_A_POLARIZATION", "Values": ["HH"]}
        ]}
        with patch.object(gcov_analysis.asf, "search", return_value=[first, second]):
            groups = gcov_analysis.fetch_area_observations(area)
        self.assertEqual(sorted(groups[(4, 9, "ASCENDING")]), ["2025-01-01"])

    def test_area_selection_uses_full_coverage_and_range_extremes(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02], [-0.02, -0.02]
        ])
        outside = observation("2025-01-05")
        outside.geometry = mapping(box(0, 0, 1, 1))
        results = [observation("2025-01-01"), observation("2025-01-02"),
                   observation("2025-01-03"), outside, observation("2025-01-06")]
        first, last = gcov_analysis.validate_dates("compare", "2025-01-01", "2025-01-05")
        with patch.object(gcov_analysis.asf, "search", return_value=results) as search:
            selected = gcov_analysis.select_area_observations(area, first, last, "compare")
        self.assertEqual([x["startTime"][:10] for x in selected], ["2025-01-01", "2025-01-03"])
        self.assertEqual(search.call_args.kwargs["start"], "2025-01-01T00:00:00Z")
        self.assertEqual(search.call_args.kwargs["end"], "2025-01-06T00:00:00Z")
        self.assertNotIn("maxResults", search.call_args.kwargs)

    def test_area_selection_rejects_mismatched_extreme_scenes(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02], [-0.02, -0.02]
        ])
        first, last = gcov_analysis.validate_dates("compare", "2025-01-01", "2025-01-03")
        with patch.object(gcov_analysis.asf, "search", return_value=[
            observation("2025-01-01"), observation("2025-01-03", frame=10)
        ]):
            with self.assertRaisesRegex(gcov_analysis.AnalysisError, "same path"):
                gcov_analysis.select_area_observations(area, first, last, "compare")

    def test_area_analysis_masks_polygon_and_supports_one_date(self):
        files = {
            "https://nisar.asf.earthdatacloud.nasa.gov/2025-01-01.h5": hdf_bytes(0.01),
            "https://nisar.asf.earthdatacloud.nasa.gov/2025-01-03.h5": hdf_bytes(0.02),
        }
        fs = SimpleNamespace(open=lambda url, *args, **kwargs: io.BytesIO(files[url]))
        polygon = [[-0.035, -0.035], [0.035, -0.035], [-0.035, 0.035], [-0.035, -0.035]]
        with patch.dict("os.environ", {"EDL_TOKEN": "test-token"}), \
             patch.object(gcov_analysis.asf, "search", return_value=[
                 observation("2025-01-01"), observation("2025-01-03")
             ]), patch.object(gcov_analysis.fsspec, "filesystem", return_value=fs):
            single = gcov_analysis.analyze_area(polygon, "single", "2025-01-01", None)
            compare = gcov_analysis.analyze_area(polygon, "compare", "2025-01-01", "2025-01-03")
        self.assertEqual(single["dates"], ["2025-01-01"])
        self.assertAlmostEqual(single["median_hv_db"], -20, places=3)
        self.assertEqual(single["valid_percent"], 100)
        image_bytes = base64.b64decode(single["images"]["single"].split(",", 1)[1])
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertTrue(np.any(np.asarray(image)[:, :, 3] == 0))
        self.assertEqual(compare["dates"], ["2025-01-01", "2025-01-03"])
        self.assertAlmostEqual(compare["median_change_db"], 3.0103, places=3)

    def test_large_area_uses_labeled_adaptive_sampling(self):
        files = {
            "https://nisar.asf.earthdatacloud.nasa.gov/2025-01-01.h5": hdf_bytes(0.01),
            "https://nisar.asf.earthdatacloud.nasa.gov/2025-01-03.h5": hdf_bytes(0.02),
        }
        fs = SimpleNamespace(open=lambda url, *args, **kwargs: io.BytesIO(files[url]))
        polygon = [[-0.039, -0.039], [0.039, -0.039], [0.039, 0.039],
                   [-0.039, 0.039], [-0.039, -0.039]]
        with patch.dict("os.environ", {"EDL_TOKEN": "test-token"}), \
             patch.object(gcov_analysis, "MAX_AREA_PIXELS", 1), \
             patch.object(gcov_analysis.asf, "search", return_value=[
                 observation("2025-01-01"), observation("2025-01-03")
             ]), patch.object(gcov_analysis.fsspec, "filesystem", return_value=fs):
            result = gcov_analysis.analyze_area(
                polygon, "compare", "2025-01-01", "2025-01-03"
            )
        self.assertEqual(result["pixel_factor"], 8)
        self.assertEqual(result["sampling"], "muestreo")
        self.assertAlmostEqual(result["median_change_db"], 3.0103, places=3)

    def test_area_is_projected_into_gcov_grid(self):
        blob = projected_hdf_bytes(0.01)
        fs = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(blob))
        polygon = [[-84.7004, 15.5996], [-84.6996, 15.5996], [-84.6996, 15.6004],
                   [-84.7004, 15.6004], [-84.7004, 15.5996]]
        scene = observation("2025-01-01")
        scene.geometry = mapping(box(-85, 15, -84, 16))
        with patch.dict("os.environ", {"EDL_TOKEN": "test-token"}), \
             patch.object(gcov_analysis.asf, "search", return_value=[scene]), \
             patch.object(gcov_analysis.fsspec, "filesystem", return_value=fs):
            result = gcov_analysis.analyze_area(polygon, "single", "2025-01-01", None)
        self.assertAlmostEqual(result["median_hv_db"], -20, places=3)
        self.assertEqual(result["valid_percent"], 100)

    def test_area_api_validates_input_and_returns_results(self):
        payload = {
            "coordinates": [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]],
            "mode": "single", "start": "2025-01-01"
        }
        with app.app.test_client() as client:
            with patch.object(app, "cached_area_analysis", return_value={"mode": "single"}) as analyze:
                response = client.post("/api/analysis", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["mode"], "single")
                self.assertEqual(analyze.call_args.args[1:], ("single", "2025-01-01", "2025-01-01"))
            invalid = dict(payload, coordinates=[[0, 0], [1, 1], [1, 0]])
            self.assertEqual(client.post("/api/analysis", json=invalid).status_code, 400)
            invalid = dict(payload, mode="compare", end="2024-01-01")
            self.assertEqual(client.post("/api/analysis", json=invalid).status_code, 400)

    def test_area_cache_reuses_results_and_separates_dates(self):
        payload = {
            "coordinates": [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]],
            "mode": "single", "start": "2025-01-01"
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FileSystemCache(directory, default_timeout=3600)
            with patch.dict(app.app.extensions["cache"], {app.analysis_cache: backend}), \
                 patch.object(app, "cached_area_observations", return_value={}), \
                 patch.object(app, "analyze_area", return_value={"mode": "single"}) as analyze, \
                 app.app.test_client() as client:
                self.assertEqual(client.post("/api/analysis", json=payload).status_code, 200)
                self.assertEqual(client.post("/api/analysis", json=payload).status_code, 200)
                self.assertEqual(analyze.call_count, 1)
                other = dict(payload, start="2025-01-02")
                self.assertEqual(client.post("/api/analysis", json=other).status_code, 200)
                self.assertEqual(analyze.call_count, 2)
                app.app.extensions["cache"][app.analysis_cache] = FileSystemCache(directory, default_timeout=3600)
                self.assertEqual(client.post("/api/analysis", json=payload).status_code, 200)
                self.assertEqual(analyze.call_count, 2)

    def test_available_dates_exclude_partial_coverage_and_incompatible_pairs(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]
        ])
        outside = observation("2025-01-04")
        outside.geometry = mapping(box(0, 0, 1, 1))
        invalid = observation("2025-01-05")
        invalid.properties["url"] = "https://example.com/data.h5"
        with patch.object(gcov_analysis.asf, "search", return_value=[
            observation("2025-01-01"), observation("2025-01-02"),
            observation("2025-01-02", frame=10), observation("2025-01-03", frame=10),
            outside, invalid,
        ]):
            dates = gcov_analysis.available_area_dates(gcov_analysis.fetch_area_observations(area))
        self.assertEqual(dates["dates"], ["2025-01-01", "2025-01-02", "2025-01-03"])
        self.assertEqual(dates["compatible"]["2025-01-01"], ["2025-01-02"])
        self.assertEqual(dates["compatible"]["2025-01-02"], ["2025-01-03"])
        self.assertEqual(dates["compatible"]["2025-01-03"], [])

    def test_available_dates_support_a_single_observation(self):
        area = gcov_analysis.validate_area([
            [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]
        ])
        with patch.object(gcov_analysis.asf, "search", return_value=[observation("2025-01-01")]):
            dates = gcov_analysis.available_area_dates(gcov_analysis.fetch_area_observations(area))
        self.assertEqual(dates, {"dates": ["2025-01-01"], "compatible": {"2025-01-01": []}})

    def test_date_endpoint_reuses_catalog_for_analysis(self):
        payload = {
            "coordinates": [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]],
            "mode": "compare", "start": "2025-01-01", "end": "2025-01-02",
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FileSystemCache(directory, default_timeout=3600)
            with patch.dict(app.app.extensions["cache"], {app.analysis_cache: backend}), \
                 patch.object(gcov_analysis.asf, "search", return_value=[
                     observation("2025-01-01"), observation("2025-01-02")
                 ]) as search, \
                 patch.object(app, "analyze_area", return_value={"mode": "compare"}) as analyze, \
                 app.app.test_client() as client:
                response = client.post("/api/analysis/dates", json={"coordinates": payload["coordinates"]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["compatible"]["2025-01-01"], ["2025-01-02"])
                self.assertEqual(client.post("/api/analysis", json=payload).status_code, 200)
                self.assertEqual(search.call_count, 1)
                self.assertEqual(len(analyze.call_args.args[4]), 1)
                self.assertEqual(client.post("/api/analysis/dates", json={"coordinates": [[0, 0]]}).status_code, 400)

    def test_date_endpoint_reports_catalog_errors(self):
        coordinates = [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, -0.02]]
        with app.app.test_client() as client, \
             patch.object(app, "cached_area_observations", side_effect=OSError("fallo de red")):
            response = client.post("/api/analysis/dates", json={"coordinates": coordinates})
            self.assertEqual(response.status_code, 502)
            self.assertIn("ASF", response.get_json()["error"])


def hdf_bytes_full(hv, hh, looks=100.0):
    """Como hdf_bytes pero con HHHH y numberOfLooks (para las categorías)."""
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as hdf:
        group = hdf.create_group(gcov_analysis.GCOV_GROUP)
        group["projection"] = 4326
        group["xCoordinates"] = np.arange(8) * 0.01 - 0.035
        group["yCoordinates"] = 0.035 - np.arange(8) * 0.01
        group["xCoordinateSpacing"] = 0.01
        group["yCoordinateSpacing"] = -0.01
        group["HVHV"] = np.full((8, 8), hv, dtype="float32")
        group["HHHH"] = np.full((8, 8), hh, dtype="float32")
        group["numberOfLooks"] = np.full((8, 8), looks, dtype="float32")
    return buffer.getvalue()


class CategoryIntegrationTests(unittest.TestCase):
    def run_area(self, files, mode, start, end):
        fs = SimpleNamespace(open=lambda url, *args, **kwargs: io.BytesIO(files[url]))
        polygon = [[-0.035, -0.035], [0.035, -0.035], [0.035, 0.035],
                   [-0.035, 0.035], [-0.035, -0.035]]
        with patch.dict("os.environ", {"EDL_TOKEN": "test-token"}), \
             patch.object(gcov_analysis.asf, "search", return_value=[
                 observation("2025-01-01"), observation("2025-01-03")
             ]), patch.object(gcov_analysis.fsspec, "filesystem", return_value=fs):
            return gcov_analysis.analyze_area(polygon, mode, start, end)

    def test_forest_loss_is_reported_with_high_confidence(self):
        base = "https://nisar.asf.earthdatacloud.nasa.gov/"
        files = {
            base + "2025-01-01.h5": hdf_bytes_full(hv=0.06, hh=0.1),
            base + "2025-01-03.h5": hdf_bytes_full(hv=0.01, hh=0.1),
        }
        result = self.run_area(files, "compare", "2025-01-01", "2025-01-03")
        self.assertEqual(result["gap_days"], 2)
        self.assertTrue(result["has_looks"])
        forest = result["categories"]["forest"]
        loss = next(row for row in forest["stats"]["classes"] if row["code"] == 1)
        self.assertAlmostEqual(loss["percent"], 100.0, places=3)
        self.assertEqual(forest["stats"]["confidence"]["high"], 100.0)
        image_bytes = base64.b64decode(forest["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.mode, "RGBA")
        self.assertGreaterEqual(len(forest["technical"]), 4)
        self.assertIn("water", result["categories"])

    def test_single_date_returns_both_categories(self):
        base = "https://nisar.asf.earthdatacloud.nasa.gov/"
        files = {base + "2025-01-01.h5": hdf_bytes_full(hv=0.06, hh=0.1),
                 base + "2025-01-03.h5": hdf_bytes_full(hv=0.06, hh=0.1)}
        result = self.run_area(files, "single", "2025-01-01", None)
        self.assertEqual(set(result["categories"]), {"water", "forest"})

    def test_files_without_hh_skip_categories(self):
        base = "https://nisar.asf.earthdatacloud.nasa.gov/"
        files = {base + "2025-01-01.h5": hdf_bytes(0.01),
                 base + "2025-01-03.h5": hdf_bytes(0.02)}
        result = self.run_area(files, "compare", "2025-01-01", "2025-01-03")
        self.assertIsNone(result["categories"])
        self.assertIn("HHHH", result["categories_note"])


if __name__ == "__main__":
    unittest.main()
