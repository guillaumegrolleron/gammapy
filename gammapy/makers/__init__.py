# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.registry import Registry
from .background import (
    AdaptiveRingBackgroundMaker,
    FoVBackgroundMaker,
    PhaseBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    RegionsFinder,
    RingBackgroundMaker,
    WobbleRegionsFinder,
)
from .core import Maker
from .map import MapDatasetMaker
from .reduce import DatasetsMaker
from .safe import SafeMaskMaker
from .spectrum import SpectrumDatasetMaker
from .events import EventDatasetMaker
from .irf_bias import IRFBiasMaker

MAKER_REGISTRY = Registry(
    [
        ReflectedRegionsBackgroundMaker,
        AdaptiveRingBackgroundMaker,
        FoVBackgroundMaker,
        PhaseBackgroundMaker,
        RingBackgroundMaker,
        SpectrumDatasetMaker,
        MapDatasetMaker,
        SafeMaskMaker,
        DatasetsMaker,
        EventDatasetMaker,
        IRFBiasMaker,
    ]
)
"""Registry of maker classes in Gammapy."""

__all__ = [
    "AdaptiveRingBackgroundMaker",
    "DatasetsMaker",
    "FoVBackgroundMaker",
    "Maker",
    "MAKER_REGISTRY",
    "MapDatasetMaker",
    "PhaseBackgroundMaker",
    "ReflectedRegionsBackgroundMaker",
    "ReflectedRegionsFinder",
    "RegionsFinder",
    "RingBackgroundMaker",
    "SafeMaskMaker",
    "SpectrumDatasetMaker",
    "WobbleRegionsFinder",
    "EventDatasetMaker",
    "IRFBiasMaker",
]
