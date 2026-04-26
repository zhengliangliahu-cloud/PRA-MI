import unittest

from autoeegencoder.diagnostics.availability import assert_method_availability
from autoeegencoder.protocols.guards import ProtocolGuard


class ProtocolGuardTests(unittest.TestCase):
    def test_p0_rejects_target_aggregate(self):
        guard = ProtocolGuard("P0")
        with self.assertRaises(ValueError):
            guard.require_target_aggregate_allowed("target covariance")

    def test_p1_allows_target_aggregate(self):
        guard = ProtocolGuard("P1")
        guard.require_target_aggregate_allowed("target covariance")
        self.assertEqual(guard.events[0]["availability"], "D1")

    def test_d0c_and_d3_cannot_enter_p0_method(self):
        with self.assertRaises(ValueError):
            assert_method_availability("P0", "D0c", "posthoc_motor_response")
        with self.assertRaises(ValueError):
            assert_method_availability("P0", "D3", "oracle_accuracy_proxy")


if __name__ == "__main__":
    unittest.main()

