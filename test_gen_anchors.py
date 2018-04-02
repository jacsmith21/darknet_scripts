import unittest

from gen_anchors import iou


class TestGenAnchors(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(0.25, iou([0.5, 0.5], [[1, 1]])[0])
        self.assertAlmostEqual(0.3333333, iou([1, 0.5], [[0.5, 1]])[0])
