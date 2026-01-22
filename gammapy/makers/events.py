# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from astropy.table import Table
from regions import PointSkyRegion
from gammapy.maps import MapAxis, Map, LabelMapAxis
from gammapy.data import Observation
from gammapy.maps import RegionGeom
from gammapy.makers import Maker, MapDatasetMaker
from gammapy.makers.utils import (
    make_edisp_kernel_map,
    make_mask_events,
    make_edisp_map,
)

ENERGY_AXIS_DEFAULT = MapAxis.from_energy_bounds(
    0.01, 300, nbin=10, per_decade=True, unit="TeV", name="energy"
)


__all__ = ["EventDatasetMaker"]


class EventDatasetMaker(Maker):
    """Make event dataset for a single IACT observation."""

    tag = "EventDatasetMaker"
    available_selection = ["events", "exposure", "background", "psf", "edisp"]

    def __init__(
        self,
        selection=None,
        debug=False,
        log=logging.getLogger(__name__),
        **maker_kwargs,
    ):
        self.log = log
        self.__debug = debug
        if selection is None:
            selection = self.available_selection

        selection = set(selection)

        if not selection.issubset(self.available_selection):
            difference = selection.difference(self.available_selection)
            raise ValueError(f"{difference} is not a valid method.")

        self.selection = selection

    @staticmethod
    def make_meta_table(observation):
        """Make info meta table.
        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation
        Returns
        -------
        meta_table: `~astropy.table.Table`
        """
        meta_table = Table()
        meta_table["TELESCOP"] = [observation.aeff.meta.get("TELESCOP", "Unknown")]
        meta_table["OBS_ID"] = [observation.obs_id]
        meta_table["RA_PNT"] = [observation.pointing.fixed_icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing.fixed_icrs.dec.deg] * u.deg

        return meta_table

    def run(self, dataset, observation):
        """Make the EventDataset.

        Parameters
        ----------
        dataset : EventDataset
            Empty EventDataset specifying the sky position from which computing the IRFs.
        observation : `~gammapy.data.Observation`
            Observation to build the EventDataset from

        Returns
        -------
        dataset : `~gammapy.datasets.EventDataset`
            EventDataset.
        """
        self.log.debug(
            f"Processing {self.__class__} to observation {observation.obs_id}"
        )
        kwargs = {"gti": observation.gti}
        if isinstance(observation, Observation):
            kwargs["meta_table"] = self.make_meta_table(observation)
        if getattr(observation, "meta"):
            kwargs["meta"] = observation.meta

        events = self.make_events(dataset._geom, observation)
        if self.__debug:
            events = events.select_time(
                [observation.gti.time_start, observation.gti.time_start + 0.3 * u.h]
            )
        kwargs["events"] = events

        axes_events = LabelMapAxis(
            labels=events.table["ENERGY"].value, name="energy", unit="TeV"
        )
        axes = (dataset.geom.axes + [axes_events]).reverse

        unbinned_geom = dataset.geom.to_image().to_cube(axes=axes)
        # geom.axes._n_spatial_axes = 0
        kwargs["geom"] = unbinned_geom

        mask_safe = Map.from_geom(unbinned_geom.drop("energy_true"), dtype=bool)
        mask_safe.data[...] = True
        kwargs["mask_safe"] = mask_safe

        if "background" in self.selection:
            raise NotImplementedError("Background not implemented yet for unbinned")

        if "psf" in self.selection:
            raise NotImplementedError("PSF not implemented yet for unbinned")

        geom_irf_for_normalization = RegionGeom.create(
            region=dataset.geom.region,
            axes=[
                ENERGY_AXIS_DEFAULT.copy(name="energy"),
                # unbinned_geom.axes["energy_true"],
            ],
        )
        kwargs["geom_normalization"] = geom_irf_for_normalization

        if "exposure" in self.selection:
            exposure = MapDatasetMaker.make_exposure(dataset.exposure.geom, observation)
            kwargs["exposure"] = exposure
            # kwargs["exposure_original_irf"] = MapDatasetMaker.make_exposure(
            #    geom_irf_for_normalization.squash(axis_name="energy"), observation
            # )

        if "edisp" in self.selection:
            if dataset.edisp.edisp_map.geom.axes[0].name.upper() == "MIGRA":
                edisp = self.make_edisp(
                    observation,
                    geom=dataset.edisp.edisp_map.geom,  # to_cube(axes=[unbinned_geom.axes["energy_true"]]),
                )
            else:
                edisp = self.make_edisp_kernel(
                    observation,
                    geom=dataset.edisp.edisp_map.geom,
                    bias=dataset.energy_reco_bias.value
                    if hasattr(dataset, "energy_reco_bias")
                    else 1.0,
                )
            # edisp.edisp_map.normalize("energy")
            kwargs["edisp"] = edisp

        # dataset = self.map_ds_maker.run(emptyMapDs, obs)
        #
        # if self.safe_mask_maker:
        #    dataset = self.safe_mask_maker.run(dataset, obs)
        #    kwargs["mask_safe"] = dataset.mask_safe
        #
        # for key in self.selection:
        #    kwargs[key] = getattr(dataset, key, None)

        kwargs["gti"] = dataset.gti

        result = dataset.__class__(name=dataset.name, **kwargs)
        result = dataset._propagate_needed_members(result)
        return result

    # def make_psf(self, observation):
    #    """Make PSF map.
    #
    #    Parameters
    #    ----------
    #    geom : `~gammapy.maps.Geom`
    #        Reference geometry.
    #    observation : `~gammapy.data.Observation`
    #        Observation container.
    #
    #    Returns
    #    -------
    #    psf : `~gammapy.irf.PSFMap`
    #        PSF map.
    #    """
    #    psf = observation.psf
    #
    #    geom = psf.psf_map.geom
    #
    #    if isinstance(psf, RecoPSFMap):
    #        return RecoPSFMap(psf.psf_map.interp_to_geom(geom))
    #    elif isinstance(psf, PSFMap):
    #        return PSFMap(psf.psf_map.interp_to_geom(geom))
    #    exposure = self.make_exposure_irf(geom.squash(axis_name="rad"), observation)
    #
    #    return make_psf_map(
    #        psf=psf,
    #        pointing=observation.get_pointing_icrs(observation.tmid),
    #        geom=geom,
    #        exposure_map=exposure,
    #    )

    #
    def make_edisp(self, observation, geom):
        """Make energy dispersion per events.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernelMap`
            Energy dispersion map.
        """
        self.log.debug(f"Making edisp map for {observation.obs_id}")
        exposure = MapDatasetMaker.make_exposure_irf(
            geom.squash(axis_name="migra"), observation
        )  # Squash over events?
        use_region_center = getattr(self, "use_region_center", True)
        # _geom = geom.to_image().to_cube(
        #    axes=geom.axes + [geom_edisp.axes["energy_true"]]
        # )
        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
            use_region_center=use_region_center,
        )

    def make_events(self, geom, observation):
        """Make counts map.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geometry.
        observation : `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        counts : `~gammapy.maps.RegionNDMap`
            Counts map.
        """
        self.log.debug(f"Making events for {observation.obs_id}")
        if geom.is_region and isinstance(geom.region, PointSkyRegion):
            mask = make_mask_events(geom, observation.rad_max, observation.events)
        else:
            mask = geom.contains(observation.events.radec)
        events = observation.events.select_row_subset(mask)
        return events

    def make_edisp_kernel(self, observation, geom, bias=1):  # , original_irf=False):
        # if original_irf:
        #    exposure = MapDatasetMaker.make_exposure_irf(
        #        geom_edisp.squash(axis_name="energy"), observation
        #    )
        # else:
        self.log.debug(f"Making edisp kernel for {observation.obs_id}")
        exposure = MapDatasetMaker.make_exposure_irf(
            geom.squash(axis_name="energy"), observation
        )  # Squash over events?
        use_region_center = getattr(self, "use_region_center", True)
        # _geom = geom.to_image().to_cube(
        #    axes=geom.axes + [geom_edisp.axes["energy_true"]]
        # )
        return make_edisp_kernel_map(
            edisp=observation.edisp,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
            use_region_center=use_region_center,
            bias=bias,
        )
