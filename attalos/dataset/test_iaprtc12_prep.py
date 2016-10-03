#!/usr/bin/env python
# -*- coding: latin1 -*-
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
Test module for iaprtc12_prep.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from attalos.dataset.iaprtc12_prep import Annotation, IAPRTC12DatasetPrep, IAPRTC12_URL, INRIA_LEAR_URL
import pytest
import requests


def test_get_id_from_path():
    """Test the extraction of a unique id from a file path."""
    uniq_id = "00/0000"

    paths = (
        "/test/test/" + uniq_id + ".jpg",
        "/test/test/" + uniq_id + ".eng",
    )

    for path in paths:
        assert uniq_id == IAPRTC12DatasetPrep.get_id_from_path(path)


def test_urls():
    """Test that the data urls are up."""
    urls = (
        IAPRTC12_URL,
        INRIA_LEAR_URL,
    )

    # Check that the data urls return 200
    for url in urls:
        ret = requests.head(url)
        assert ret.status_code == requests.codes.ok


def test_annotation():
    """Test that the Annotation class correctly converts XML."""

    xml = """
        <DOC>
        <DOCNO>annotations/02/2808.eng</DOCNO>
        <TITLE>In the Tafí del Valle in the province Tucumán</TITLE>
        <DESCRIPTION>a man on a dry slope with a few dry tussocks; a lake and a brown, bald mountain landscape in the background;</DESCRIPTION>
        <NOTES>Tafí del Valle is in the west of the province Tucumán, in the northeast of Argentina;  it is a weekend and a summer destination for the Tucumanos;</NOTES>
        <LOCATION>Tucumán, Argentina</LOCATION>
        <DATE>January 2002</DATE>
        <IMAGE>images/02/2808.jpg</IMAGE>
        <THUMBNAIL>thumbnails/02/2808.jpg</THUMBNAIL>
        </DOC>
        """

    title = u"In the Tafí del Valle in the province Tucumán"
    description = "a man on a dry slope with a few dry tussocks; a lake and a brown, bald mountain landscape in the background;"
    uniq_id = "02/2808"

    # Test the Annotation Class
    a = Annotation(xml)
    assert title == a.title
    assert description == a.description
    assert uniq_id == a.uniq_id

# TODO: Add tests for the rest of the class. These require the raw data (2
# gigs) so some work around is needed.
