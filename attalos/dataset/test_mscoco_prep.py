#!/usr/bin/env python
#
#   Copyright (c) 2016 In-Q-Tel, Inc, All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Test module for mscoco_prep.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from attalos.dataset.mscoco_prep import TRAIN_VAL_INSTANCES_2014_URL, TRAIN_VAL_IMAGE_CAPTIONS_2014_URL, TRAIN_IMAGE_2014_URL

import pytest
import requests


def test_urls():
    """Test that the data urls are up."""
    urls = (
        TRAIN_VAL_INSTANCES_2014_URL,
        TRAIN_VAL_IMAGE_CAPTIONS_2014_URL,
        TRAIN_IMAGE_2014_URL
    )

    # Check that the data urls return 200
    for url in urls:
        ret = requests.head(url)
        assert ret.status_code == requests.codes.ok

# TODO: Add tests for the rest of the class. These require the raw data (2
# gigs) so some work around is needed.
