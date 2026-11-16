import unittest

import numpy as np

import parameters as P


def lin(db):
    return 10 ** (np.asarray(db, dtype="float64") / 10)


def scene(hh_db=-8.0, hv_db=-11.5, shape=(20, 20)):
    return np.full(shape, hh_db, "float32"), np.full(shape, hv_db, "float32")


class IndexTests(unittest.TestCase):
    def test_rvi_and_rfdi_known_values(self):
        one = np.array([[1.0]])
        self.assertAlmostEqual(float(P.rvi(one, one)[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(P.rfdi(one, one)[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(P.rfdi(np.array([[3.0]]), one)[0, 0]), 0.5, places=5)

    def test_to_db_masks_invalid(self):
        out = P.to_db(np.array([1.0, 0.0, -1.0, np.nan, 10.0]))
        self.assertAlmostEqual(float(out[0]), 0.0, places=5)
        self.assertTrue(np.isnan(out[1:4]).all())
        self.assertAlmostEqual(float(out[4]), 10.0, places=4)

    def test_noise_falls_with_looks(self):
        self.assertAlmostEqual(float(P.noise_db(100)), 0.434, places=3)
        self.assertAlmostEqual(float(P.noise_db(None)), 4.34, places=2)
        self.assertGreater(float(P.noise_db(1)), float(P.noise_db(16)))


class ConfidenceTests(unittest.TestCase):
    def test_levels(self):
        conf = P.confidence_from_z(np.array([0.5, 2.0, 3.9, 4.0, 10.0]))
        self.assertEqual(conf.tolist(), [0, 1, 1, 2, 2])

    def test_gap_caps(self):
        z = np.array([10.0])
        self.assertEqual(P.confidence_from_z(z, 12).tolist(), [2])
        self.assertEqual(P.confidence_from_z(z, 36).tolist(), [1])
        self.assertEqual(P.confidence_from_z(z, 90).tolist(), [0])


class WaterTests(unittest.TestCase):
    def test_new_water_and_lost_water(self):
        hh1, hv1 = scene()
        hh2, hv2 = hh1.copy(), hv1.copy()
        hh2[:5, :5], hv2[:5, :5] = -24, -30           # tierra -> agua
        hh1[10:15, 10:15], hv1[10:15, 10:15] = -24, -30   # agua -> tierra
        looks = np.full(hh1.shape, 100.0)
        cls, conf = P.water_change(hh1, hv1, hh2, hv2, looks, looks, gap_days=12)
        self.assertTrue((cls[:5, :5] == 2).all())
        self.assertTrue((cls[10:15, 10:15] == 3).all())
        self.assertTrue((cls[5:10, 5:10] == 0).all())
        self.assertTrue((conf[:5, :5] == P.HIGH).all())
        self.assertTrue((conf[5:10, 5:10] == P.NODATA).all())   # sin cambio: sin fiabilidad

    def test_stable_water(self):
        hh, hv = scene(-24, -30)
        cls, _ = P.water_change(hh, hv, hh, hv)
        self.assertTrue((cls == 1).all())

    def test_few_looks_lowers_confidence(self):
        hh1, hv1 = scene()
        hh2, hv2 = scene(-24, -30)
        cls, conf = P.water_change(hh1, hv1, hh2, hv2, looks1=None, looks2=None)
        self.assertTrue((cls == 2).all())
        self.assertTrue((conf <= P.MEDIUM).all())

    def test_long_gap_never_high(self):
        hh1, hv1 = scene()
        hh2, hv2 = scene(-24, -30)
        looks = np.full(hh1.shape, 100.0)
        _, conf = P.water_change(hh1, hv1, hh2, hv2, looks, looks, gap_days=90)
        self.assertTrue((conf == P.LOW).all())

    def test_flooded_vegetation_candidate(self):
        hh1, hv1 = scene()
        hh2, hv2 = hh1.copy(), hv1.copy()
        hh2[:4, :4] += 5.0                      # HH sube, HV igual
        cls, _ = P.water_change(hh1, hv1, hh2, hv2)
        self.assertTrue((cls[:4, :4] == 4).all())
        self.assertTrue((cls[4:, 4:] == 0).all())

    def test_nodata_propagates(self):
        hh1, hv1 = scene()
        hh2, hv2 = hh1.copy(), hv1.copy()
        hv2[0, 0] = np.nan
        cls, conf = P.water_change(hh1, hv1, hh2, hv2)
        self.assertEqual(cls[0, 0], P.NODATA)
        self.assertEqual(conf[0, 0], P.NODATA)

    def test_single_date_state(self):
        hh, hv = scene()
        hh[:5, :5], hv[:5, :5] = -24, -30
        cls, conf = P.water_state(hh, hv, np.full(hh.shape, 100.0))
        self.assertTrue((cls[:5, :5] == 1).all())
        self.assertTrue((cls[5:, 5:] == 0).all())
        self.assertTrue((conf[:5, :5] == P.HIGH).all())


class ForestTests(unittest.TestCase):
    def test_loss_disturbance_and_rise(self):
        hh1, hv1 = scene()
        hh2, hv2 = hh1.copy(), hv1.copy()
        hv2[:5, :5] -= 6.0        # pérdida
        hv2[:5, 10:15] -= 2.0     # perturbación leve
        hv2[10:15, :5] += 4.0     # aumento
        looks = np.full(hh1.shape, 100.0)
        cls, conf = P.forest_change(hh1, hv1, hh2, hv2, looks, looks, 12)
        self.assertTrue((cls[:5, :5] == 1).all())
        self.assertTrue((cls[:5, 10:15] == 2).all())
        self.assertTrue((cls[10:15, :5] == 3).all())
        self.assertTrue((cls[15:, 15:] == 0).all())
        self.assertTrue((conf[:5, :5] == P.HIGH).all())

    def test_small_rise_in_dense_forest_stays_stable(self):
        hh1, hv1 = scene()
        hv2 = hv1 + 2.0
        cls, _ = P.forest_change(hh1, hv1, hh1, hv2)
        self.assertTrue((cls == 0).all())

    def test_water_is_not_counted_as_forest_loss(self):
        hh1, hv1 = scene()
        hh2, hv2 = scene(-24, -30)
        cls, _ = P.forest_change(hh1, hv1, hh2, hv2)
        self.assertTrue((cls == 5).all())

    def test_non_dense_surfaces_are_not_forest_loss(self):
        hh1, hv1 = scene(-10, -21)
        hv2 = hv1 - 5.0
        cls, _ = P.forest_change(hh1, hv1, hh1, hv2)
        self.assertFalse((cls == 1).any())

    def test_single_date_bins(self):
        hv = np.array([[-25.0, -19.0, -14.0, -10.0]], "float32")
        cls, _ = P.forest_state(hv)
        self.assertEqual(cls.tolist(), [[0, 1, 2, 3]])


class OutputTests(unittest.TestCase):
    def layers(self, compare=True):
        hh, hv = lin(np.full((12, 12), -8.0)), lin(np.full((12, 12), -11.5))
        d = {"hh": hh, "hv": hv, "looks": np.full((12, 12), 100.0)}
        if compare:
            hv2 = hv.copy()
            hv2[:4, :4] = lin(-20.0)
            d.update({"hh2": hh, "hv2": hv2, "looks2": np.full((12, 12), 100.0)})
        return d

    def test_build_compare_has_both_categories(self):
        out = P.build_categories(self.layers(), "compare", gap_days=12, pixel_area_km2=0.0064)
        self.assertEqual(set(out), {"water", "forest"})
        forest = out["forest"]["stats"]
        loss = next(r for r in forest["classes"] if r["code"] == 1)
        self.assertAlmostEqual(loss["percent"], 16 / 144 * 100, places=3)
        self.assertAlmostEqual(loss["km2"], 16 * 0.0064, places=6)
        self.assertAlmostEqual(sum(r["percent"] for r in forest["classes"]), 100.0, places=3)
        self.assertEqual(forest["confidence"]["high"], 100.0)
        self.assertTrue(any("reliability" in t.lower() for t in out["forest"]["summary"]))

    def test_build_single(self):
        out = P.build_categories(self.layers(False), "single")
        self.assertEqual(out["water"]["classes"].shape, (12, 12))
        self.assertIsNone(out["forest"]["stats"]["confidence"])

    def test_missing_looks_still_works(self):
        layers = self.layers()
        layers.pop("looks"), layers.pop("looks2")
        out = P.build_categories(layers, "compare", gap_days=12)
        self.assertLessEqual(out["forest"]["stats"]["confidence"]["high"], 100.0)

    def test_rgba_fades_low_confidence(self):
        cls = np.array([[1, 1, 0]], "uint8")
        conf = np.array([[P.HIGH, P.LOW, P.NODATA]], "uint8")
        rgba = P.classes_to_rgba(cls, conf, P.FOREST_COMPARE)
        self.assertEqual(rgba[0, 0, 3], 255)
        self.assertEqual(rgba[0, 1, 3], 80)
        self.assertEqual(rgba[0, 2, 3], 255)
        self.assertEqual(tuple(rgba[0, 0, :3]), (0xD5, 0x5E, 0x00))

    def test_nodata_pixels_are_transparent(self):
        cls = np.array([[P.NODATA]], "uint8")
        rgba = P.classes_to_rgba(cls, cls, P.WATER_COMPARE)
        self.assertEqual(rgba[0, 0, 3], 0)


if __name__ == "__main__":
    unittest.main()
