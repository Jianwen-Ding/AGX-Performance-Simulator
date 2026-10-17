#!/usr/bin/env python3
"""Fail-closed schema tests for retained SR replay; no simulator or GPU needed."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sr_check


class SchemaTests(unittest.TestCase):
    def setUp(self):
        counts = {'uniform': 64, 'depth1': 32, 'depth2': 16, 'depth3': 8, 'quad': 16, 'quad-depth2': 8}
        self.fixture = {'patterns': {name: {'active_lanes': list(range(n)),
            'values': {str(s): [0] * n for s in range(256)}} for name, n in counts.items()}}

    def rejects(self, fixture):
        with tempfile.TemporaryDirectory(prefix='agx-sr-schema-') as temp:
            directory = Path(temp)
            with gzip.open(directory / 'hardware.json.gz', 'wt') as stream:
                json.dump(fixture, stream)
            with patch.object(sr_check, 'D', directory), patch.object(sr_check.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    sr_check.main()
                run.assert_not_called()

    def test_empty_or_missing_patterns(self):
        self.rejects({'patterns': {}})
        del self.fixture['patterns']['uniform']
        self.rejects(self.fixture)

    def test_active_lanes_cannot_be_truncated_duplicated_or_out_of_range(self):
        for lanes in [[], [0] * 64, list(range(1, 65))]:
            f = copy.deepcopy(self.fixture)
            f['patterns']['uniform']['active_lanes'] = lanes
            self.rejects(f)
        f = copy.deepcopy(self.fixture)
        f['patterns']['uniform']['active_lanes'] = list(range(32))
        f['patterns']['uniform']['values'] = {str(s): [0] * 32 for s in range(256)}
        self.rejects(f) # shrinking expected lane values too must not hide truncation

    def test_missing_selectors_or_short_expected_values(self):
        del self.fixture['patterns']['depth1']['values']['255']
        self.rejects(self.fixture)
        f = copy.deepcopy(self.fixture)
        f['patterns']['depth1']['values']['255'] = [0] * 32
        f['patterns']['depth2']['values']['7'] = []
        self.rejects(f)


if __name__ == '__main__':
    unittest.main()
