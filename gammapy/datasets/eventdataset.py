# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import logging
from astropy import units as u
from astropy.utils import lazyproperty
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    DatasetModels,
    FoVBackgroundModel,
    NormBackgroundSpectralModel,
    Models,
    GaussianPrior,
)
from gammapy.utils.scripts import make_name
from gammapy.utils.fits import LazyFitsData, HDULocation
from gammapy.utils.integrate import integrate_histogram
from gammapy.irf import EDispMap, EDispKernel, EDispKernelMap, PSFMap, RecoPSFMap
from gammapy.maps import Map, MapAxes, LabelMapAxis, MapAxis
from gammapy.data import GTI
from .unbinned_evaluator import UnbinnedEvaluator
from .core import Dataset
from .spectrum import PlotMixin
from .map import BINSZ_IRF_DEFAULT, RAD_AXIS_DEFAULT, MIGRA_AXIS_DEFAULT

log = logging.getLogger(__name__)

EVALUATION_MODE = "local"
USE_NPRED_CACHE = True

ENERGY_AXIS_DEFAULT = MapAxis.from_energy_bounds(
    0.01, 300, nbin=10, per_decade=True, unit="TeV", name="energy"
)


class EventDataset(Dataset, PlotMixin):
    """ """

    tag = "EventDataset"
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
    background = LazyFitsData(cache=True)
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    _mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = [
        "background",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "_mask_safe",
    ]

    gti = None
    meta_table = None

    def __init__(
        self,
        events=None,
        geom=None,
        geom_normalization=None,
        models=None,
        background=None,
        exposure=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        name=None,
        reference_time="2000-01-01",
        gti=None,
        meta=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}
        # self.position = position
        self.geom = geom
        self.geom_normalization = geom_normalization

        self.events = events
        self.exposure = exposure
        if not isinstance(self, EventDatasetOnOff):
            self.background = background
        # self._background_cached = None
        # self._background_parameters_cached = None

        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

        self.reference_time = reference_time
        self.gti = gti
        self.models = models
        self.meta_table = meta_table

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )
        self.psf = psf

        if edisp is not None and not isinstance(
            edisp, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )
        # if edisp_e_reco_binned is not None and not isinstance(
        #    edisp_e_reco_binned, (EDispMap, EDispKernelMap, HDULocation)
        # ):
        #    raise ValueError(
        #        "'edisp_e_reco_binned' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
        #        f"object, got `{type(edisp_e_reco_binned)}` instead."
        #    )

        self.edisp = edisp
        self.meta = meta
        # self.edisp_e_reco_binned = edisp_e_reco_binned
        # self.exposure_original_irf = exposure_original_irf

    @property
    def mask_safe(self):
        """Getter for mask_safe"""
        return self._mask_safe

    @mask_safe.setter
    def mask_safe(self, value):
        """Setter for mask_safe with custom processing"""
        self._mask_safe = value

    @lazyproperty
    def __edisp_e_reco_binned(self):
        if isinstance(self.edisp, (EDispKernelMap, EDispKernel)):
            if hasattr(self, "energy_reco_bias") and self.energy_reco_bias is not None:
                log.warning(
                    "Energy bias is not applied to edisp_e_reco_binned because\
                        energy dispersion is already a kernel. Please Check your EventDatasetMaker."
                )
            return self.edisp
        else:
            energy_axes = self.geom_normalization.axes["energy"]
            bias = (
                self.energy_reco_bias.value
                if hasattr(self, "energy_reco_bias")
                and self.energy_reco_bias is not None
                else 1.0
            )
            self._bias_cached = bias
            return self.edisp.to_edisp_kernel_map(energy_axes, bias=bias)

    @property
    def edisp_e_reco_binned(self):
        """Energy dispersion kernel map for binned energy axis."""
        if self._energy_reco_bias_has_changed:
            if "__edisp_e_reco_binned" in self.__dict__:
                del self.__dict__["__edisp_e_reco_binned"]
        return self.__edisp_e_reco_binned

    @lazyproperty
    def __edisp_kernel_unbinned(self):
        """Energy dispersion kernel map for unbinned energy axis."""
        edisp_e_reco_binned = self.edisp_e_reco_binned
        axes = self.geom.axes + [edisp_e_reco_binned.edisp_map.geom.axes["energy_true"]]
        new_geom = self.geom.to_image().to_cube(axes=axes)
        differencial_edisp_map_e_reco_binned = (
            edisp_e_reco_binned.edisp_map.divide_bin_width("energy")
        )
        edisp_map_iterpolated = differencial_edisp_map_e_reco_binned.interp_to_geom(
            geom=new_geom
        )
        edisp = EDispKernelMap(
            edisp_kernel_map=edisp_map_iterpolated,
            exposure_map=edisp_e_reco_binned.exposure_map,
        )
        normalization = integrate_histogram(
            differencial_edisp_map_e_reco_binned.quantity,
            edisp_e_reco_binned.edisp_map.geom.axes["energy"].edges,
            self.geom.axes["energy"].center.min(),
            self.geom.axes["energy"].center.max(),
            axis=differencial_edisp_map_e_reco_binned.geom.axes.index_data("energy"),
        )[0]
        normalization_inv = np.nan_to_num(normalization**-1, posinf=1)
        edisp.edisp_map.quantity = (
            np.einsum("trxy,txy->trxy", edisp.edisp_map.data, normalization_inv)
            * edisp.edisp_map.unit
            * normalization_inv.unit
        )
        return edisp

    @property
    def edisp_kernel_unbinned(self):
        """Energy dispersion kernel map for unbinned energy axis."""
        if self._energy_reco_bias_has_changed:
            if "__edisp_kernel_unbinned" in self.__dict__:
                del self.__dict__["__edisp_kernel_unbinned"]
        return self.__edisp_kernel_unbinned

    @property
    def _geom(self):
        """Main analysis geometry."""
        return self.geom

    def __str__(self):
        pass

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators

    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models

    @models.setter
    def models(self, models):
        """Models setter."""
        self._evaluators = {}
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = UnbinnedEvaluator(
                        model=model,
                        geom=self.geom,
                        geom_normalization=self.geom_normalization,
                        psf=self.psf,
                        edisp=self.edisp_kernel_unbinned,
                        edisp_e_reco_binned=self.edisp_e_reco_binned,
                        exposure=self.exposure,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                        bias_parameters=self.bias_parameters
                        if hasattr(self, "bias_parameters")
                        else None,
                    )
                    self._evaluators[model.name] = evaluator
        self._models = models

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = value

    @property
    def mask_event(self):
        """Entry for each event whether it is inside the mask or not"""
        if self.mask is None:
            return np.ones(len(self.events.table), dtype=bool)
        coords = self.events.map_coord(self.mask.geom)
        return self.mask.get_by_coord(coords) == 1

    def info_dict(self):
        pass

    def stat_array(self):
        pass

    def peek(self, figsize=(16, 4)):
        """Quick-look summary plots.

        This method creates a figure displaying the elements of your `EventDataset`.
        For example:

        * Exposure map
        * Energy dispersion matrix at the geometry center

        Parameters
        ----------
        figsize : tuple
            Size of the figure. Default is (16, 4).

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.set_title("Exposure")
        self.exposure.plot(ax1, ls="-", markersize=0, xerr=None)

        ax2.set_title("Energy Dispersion")

        if self.edisp_e_reco_binned is not None:
            self.edisp_e_reco_binned.plot_matrix(ax=ax2, add_cbar=True)

    @property
    def events_safe(self):
        return self.events.select_row_subset(self.mask_safe.data.astype(bool).flatten())

    @property
    def events_fit(self):
        return self.events.select_row_subset(self.mask_safe.data.astype(bool).flatten())

    @property
    def mask_image(self):
        """Reduced mask."""
        if self.mask is None:
            mask = Map.from_geom(self._geom.to_image(), dtype=bool)
            mask.data |= True
            return mask

        return self.mask.reduce_over_axes(func=np.logical_or)

    @property
    def geoms(self):  # DUPLICATE OF MAPDATASET
        """Map geometries.

        Returns
        -------
        geoms : dict
            Dictionary of map geometries involved in the dataset.
        """
        geoms = {}

        geoms["geom"] = self._geom
        geoms["geom_normalization"] = self.geom_normalization

        if self.exposure:
            geoms["geom_exposure"] = self.exposure.geom

        if self.psf:
            geoms["geom_psf"] = self.psf.psf_map.geom

        if self.edisp:
            geoms["geom_edisp"] = self.edisp.edisp_map.geom

        return geoms

    def to_masked(self, name=None, nan_to_num=True):
        """Return masked dataset.

        Parameters
        ----------
        name : str, optional
            Name of the masked dataset. Default is None.
        nan_to_num : bool
            Non-finite values are replaced by zero if True. Default is True.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Masked dataset.
        """
        dataset = self.__class__.from_geoms(**self.geoms, name=name)
        dataset.stack(self, nan_to_num=nan_to_num)
        return dataset

    def stack(self, other, nan_to_num=True):
        r"""Stack another dataset in place. The original dataset is modified.

        Safe mask is applied to the other dataset to compute the stacked counts data.
        Counts outside the safe mask are lost.

        Note that the masking is not applied to the current dataset. If masking needs
        to be applied to it, use `~gammapy.MapDataset.to_masked()` first.

        The stacking of 2 datasets is implemented as follows. Here, :math:`k`
        denotes a bin in reconstructed energy and :math:`j = {1,2}` is the dataset number.

        The ``mask_safe`` of each dataset is defined as:

        .. math::

            \epsilon_{jk} =\left\{\begin{array}{cl} 1, &
            \mbox{if bin k is inside the thresholds}\\ 0, &
            \mbox{otherwise} \end{array}\right.

        Then the total ``counts`` and model background ``bkg`` are computed according to:

        .. math::

            \overline{\mathrm{n_{on}}}_k =  \mathrm{n_{on}}_{1k} \cdot \epsilon_{1k} +
             \mathrm{n_{on}}_{2k} \cdot \epsilon_{2k}.

            \overline{bkg}_k = bkg_{1k} \cdot \epsilon_{1k} +
             bkg_{2k} \cdot \epsilon_{2k}.

        The stacked ``safe_mask`` is then:

        .. math::

            \overline{\epsilon_k} = \epsilon_{1k} OR \epsilon_{2k}.

        For details, see :ref:`stack`.

        Parameters
        ----------
        other : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            Map dataset to be stacked with this one. If other is an on-off
            dataset alpha * counts_off is used as a background model.
        nan_to_num : bool
            Non-finite values are replaced by zero if True. Default is True.

        """

        if self.exposure and other.exposure:
            self.exposure.stack(
                other.exposure, weights=other.mask_safe_image, nan_to_num=nan_to_num
            )
            # TODO: check whether this can be improved e.g. handling this in GTI

            if "livetime" in other.exposure.meta and np.any(other.mask_safe_image):
                if "livetime" in self.exposure.meta:
                    self.exposure.meta["livetime"] += other.exposure.meta["livetime"]
                else:
                    self.exposure.meta["livetime"] = other.exposure.meta[
                        "livetime"
                    ].copy()

        if self.stat_type == "cash":
            if self.background and other.background:
                background = self.npred_background()
                background.stack(
                    other.npred_background(),
                    weights=other.mask_safe,
                    nan_to_num=nan_to_num,
                )
                self.background = background

        if self.psf and other.psf:
            self.psf.stack(other.psf, weights=other.mask_safe_psf)

        if self.edisp and other.edisp:
            self.edisp.stack(other.edisp, weights=other.mask_safe_edisp)

        if self.mask_safe and other.mask_safe:
            self.mask_safe.stack(other.mask_safe)

        if self.mask_fit and other.mask_fit:
            self.mask_fit.stack(other.mask_fit)
        elif other.mask_fit:
            self.mask_fit = other.mask_fit.copy()

        if self.gti and other.gti:
            self.gti.stack(other.gti)
            self.gti = self.gti.union()

        if self.meta_table and other.meta_table:
            self.meta_table = hstack_columns(self.meta_table, other.meta_table)
        elif other.meta_table:
            self.meta_table = other.meta_table.copy()

        if self.meta and other.meta:
            self.meta.stack(other.meta)

        if hasattr(other, "bias_parameters"):
            for parameter in other.bias_parameters:
                if parameter.name in VALID_BIAS_PARAMETER_NAMES:
                    bias_parameter = getattr(self, parameter.name, None)
                    if bias_parameter is None:
                        setattr(self, parameter.name, parameter.copy())
                    else:
                        logging.debug(
                            f"Bias parameter {parameter.name} already exists in the dataset. Keeping the existing one."
                        )

    @classmethod
    def create(
        cls,
        geom,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=BINSZ_IRF_DEFAULT,
        reco_psf=False,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
        **kwargs,
    ):
        """Create an empty event dataset."""
        geoms = create_event_dataset_geoms(
            geom=geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            rad_axis=rad_axis,
            binsz_irf=binsz_irf,
            reco_psf=reco_psf,
        )
        kwargs.update(geoms)
        return cls.from_geoms(
            name=name, reference_time=reference_time, meta_table=meta_table, **kwargs
        )

    @classmethod
    def from_geoms(
        cls,
        geom,
        geom_exposure=None,
        geom_psf=None,
        geom_edisp=None,
        reference_time="2000-01-01",
        name=None,
        **kwargs,
    ):
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name

        if geom_exposure:
            kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")

        if geom_edisp:
            if "energy" in geom_edisp.axes.names:
                kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
            else:
                kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

        if geom_psf:
            if "energy_true" in geom_psf.axes.names:
                kwargs["psf"] = PSFMap.from_geom(geom_psf)
            elif "energy" in geom_psf.axes.names:
                kwargs["psf"] = RecoPSFMap.from_geom(geom_psf)

        kwargs.setdefault(
            "gti", GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        )
        kwargs["mask_safe"] = Map.from_geom(geom, unit="", dtype=bool)
        return cls(geom=geom, **kwargs)

    def npred(self):
        """Total predicted source and background counts.

        Returns
        -------
        npred : `Map`
            Total predicted counts.
        """
        npred_total = self.npred_signal()

        npred_total += self.npred_background()
        npred_total.data[npred_total.data < 0.0] = 0
        return npred_total

    def npred_background(self):
        raise NotImplementedError(
            "The method npred_background() is not implemented for EventDataset."
        )

    def npred_signal(self, model_names=None, stack=True):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        Parameters
        ----------
        model_names : list of str
            List of name of  SkyModel for which to compute the npred.
            If none, all the SkyModel predicted counts are computed.
        stack : bool
            Whether to stack the npred maps upon each other.

        Returns
        -------
        npred_sig : `gammapy.maps.Map`
            Map of the predicted signal counts.
        """
        npred_total = Map.from_geom(self._geom.squash("energy"), dtype=float)

        evaluators = self.evaluators
        if model_names is not None:
            if isinstance(model_names, str):
                model_names = [model_names]
            evaluators = {name: self.evaluators[name] for name in model_names}

        npred_list = []
        labels = []
        for evaluator_name, evaluator in evaluators.items():
            if evaluator.needs_update:
                evaluator.update(
                    exposure=self.exposure,
                    psf=self.psf,
                    edisp=self.edisp_kernel_unbinned,
                    edisp_e_reco_binned=self.edisp_e_reco_binned,
                    geom=self._geom,
                    mask=self.mask,
                )
            if evaluator.contributes:
                npred = evaluator.compute_npred()
                shape_expected = list(npred.data.shape)
                shape_expected[npred.geom.axes.index_data("energy")] = -1
                npred.data = npred.data.reshape(shape_expected)
                bin_width = npred.geom.axes["energy"].bin_width.value.reshape(
                    shape_expected
                )
                data = integrate_histogram(
                    npred.data / bin_width,
                    npred.geom.axes["energy"].edges.value,
                    self.events_safe.energy.value.min(),
                    self.events_safe.energy.value.max(),
                )
                npred = Map.from_geom(npred.geom.squash("energy"), data=data, unit="")

                if stack:
                    npred_total.stack(npred)
                else:
                    npred_geom = Map.from_geom(self._geom, dtype=float)
                    npred_geom.stack(npred)
                    labels.append(evaluator_name)
                    npred_list.append(npred_geom)
                if not USE_NPRED_CACHE:
                    evaluator.reset_cache_properties()

        if npred_list != []:
            label_axis = LabelMapAxis(labels=labels, name="models")
            npred_total = Map.from_stack(npred_list, axis=label_axis)

        return npred_total


def create_event_dataset_geoms(
    geom,
    energy_axis_true=None,
    migra_axis=None,
    rad_axis=None,
    binsz_irf=BINSZ_IRF_DEFAULT,
    reco_psf=False,
):
    """Create geometries needed for event dataset.
    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom` or `~gammapy.maps.RegionGeom`
        Reference geometry.
    energy_axis_true : `~gammapy.maps.MapAxis`
        True energy axis.
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis.
    rad_axis : `~gammapy.maps.MapAxis`
        Offset axis.
    binsz_irf : float
        Bin size for IRF maps in deg.
    reco_psf : bool
        Use reconstructed energy axis for PSF map.
    Returns
    -------
    dict
        Dictionary of geometries.
    """
    if rad_axis is None:
        rad_axis = RAD_AXIS_DEFAULT
    if migra_axis is None:
        migra_axis = MIGRA_AXIS_DEFAULT

    if energy_axis_true is not None:
        energy_axis_true.assert_name("energy_true")
    else:
        energy_axis_true = geom.axes["energy_true"].copy(name="energy_true")

    external_axes = geom.axes.drop("energy_true")
    geom_image = geom.to_image()
    geom_exposure = geom_image.to_cube(MapAxes([energy_axis_true]) + external_axes)
    geom_irf = geom_image.to_binsz(binsz=binsz_irf)

    if reco_psf:
        raise NotImplementedError(
            "PSF map with reco energy axis not implemented yet for event dataset."
        )
    geom_psf = geom_irf.to_cube(MapAxes([rad_axis, energy_axis_true]) + external_axes)
    geom_edisp = geom_irf.to_cube(
        MapAxes([migra_axis, energy_axis_true]) + external_axes
    )
    return {
        "geom": geom,
        "geom_exposure": geom_exposure,
        "geom_psf": geom_psf,
        "geom_edisp": geom_edisp,
    }


class EventDatasetOnOff(EventDataset):
    """Event dataset for on-off analysis."""

    def __init__(
        self,
        events=None,
        events_off=None,
        acceptance=None,
        acceptance_off=None,
        stat_type="unbinned_onoff",
        *args,
        **kwargs,
    ):
        super().__init__(events=events, *args, **kwargs)
        self.events_off = events_off
        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self.stat_type = stat_type
        self.background_model = NormBackgroundSpectralModel(
            energy_events=self.events_off.energy,
            alpha=self.alpha.data.flatten(),
        )

        bkg_model = FoVBackgroundModel(
            dataset_name=self.name, spectral_model=self.background_model
        )
        if self.models is not None:
            _models = Models(self.models)
            _models.append(bkg_model)
        else:
            _models = [bkg_model]
        self.models = DatasetModels(_models)

    def to_masked(self, name=None, nan_to_num=True):
        """Return masked dataset.

        Parameters
        ----------
        name : str, optional
            Name of the masked dataset. Default is None.
        nan_to_num : bool
            Non-finite values are replaced by zero if True. Default is True.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Masked dataset.
        """
        dataset = self.__class__.from_geoms(
            **self.geoms,
            name=name,
            events=self.events,
            events_off=self.events_off,
            acceptance=self.acceptance,
            acceptance_off=self.acceptance_off,
            stat_type=self.stat_type,
        )
        dataset.stack(self, nan_to_num=nan_to_num)
        return dataset

    @classmethod
    def from_eventdataset(
        cls,
        dataset,
        acceptance,
        acceptance_off,
        events_off,
        name,
    ):
        if np.isscalar(acceptance):
            acceptance = Map.from_geom(dataset._geom, data=acceptance)
        if np.isscalar(acceptance_off):
            acceptance_off = Map.from_geom(dataset._geom, data=acceptance_off)

        out = cls(
            events=dataset.events,
            geom=dataset.geom,
            geom_normalization=dataset.geom_normalization,
            models=dataset.models,
            exposure=dataset.exposure,
            psf=dataset.psf,
            edisp=dataset.edisp,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            meta_table=dataset.meta_table,
            name=name,
            reference_time=dataset.reference_time,
            gti=dataset.gti,
            meta=dataset.meta,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            events_off=events_off,
        )
        out = dataset._propagate_needed_members(out)
        return out

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions.

        See :ref:`wstat`.

        Returns
        -------
        alpha : `Map`
            Alpha map.
        """
        # WARNING : ALPHA IS NOT BINNED AND WE HAVE TO CHANGE THAT
        # log.warning("Alpha is not binned, this should be changed.")
        with np.errstate(invalid="ignore", divide="ignore"):
            data = self.acceptance.quantity / self.acceptance_off.quantity
        if hasattr(self, "background_bias"):
            data = self.background_bias.value * data
        data = np.nan_to_num(data)

        return Map.from_geom(self.acceptance.geom, data=data.to_value(""), unit="")

    def npred_background(self):
        """Predicted background total count from the background model interpolated on off counts.

        See :ref:`wstat`.

        Returns
        -------
        npred_background : `Map`
            Predicted background counts.
        """
        # x = np.logspace(
        #    np.log10(self.events_safe.energy.min().value),
        #    np.log10(self.events_safe.energy.max().value),
        #    100
        # ) * self.events_safe.energy.unit
        # mu_bkg = simpson(
        #    self.background_model(x),
        #    x
        # )
        if hasattr(self, "background_bias"):
            background_bias = self.background_bias.value
        else:
            background_bias = 1.0
        mu_bkg = background_bias * self.background_model.integral(
            energy_min=self.events_safe.energy.min(),
            energy_max=self.events_safe.energy.max(),
        )
        mu_bkg = np.nan_to_num(mu_bkg)
        return Map.from_geom(
            geom=self.geom.squash("energy"), data=mu_bkg.value, unit=mu_bkg.unit
        )

    @property
    def background(self):
        """Computed as alpha * n_off.

        See :ref:`wstat`.

        Returns
        -------
        background : `Map`
            Background map.
        """
        if self.background_model is None:
            return None
        background = self.background_model(self.events.energy)
        return Map.from_geom(self._geom, data=background.value, unit=background.unit)

    def signal_pdf(
        self, energy_min=None, energy_max=None, return_normalization_factor=False
    ):
        """Signal PDF evaluated at the event energies.

        Returns
        -------
        signal_pdf : `Map`
            Signal PDF map.
        """
        if not self.evaluators:
            raise ValueError("No model defined.")
        signal_pdf = None
        if return_normalization_factor:
            normalization_factor = []
        n = 0
        for evaluator in self.evaluators.values():
            if evaluator.needs_update:
                evaluator.update(
                    exposure=self.exposure,
                    psf=self.psf,
                    edisp=self.edisp_kernel_unbinned,
                    edisp_e_reco_binned=self.edisp_e_reco_binned,
                    geom=self._geom,
                    mask=self.mask,
                )

            if evaluator.contributes:
                n += 1
                _output = evaluator.compute_signal_pdf(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    return_normalization_factor=return_normalization_factor,
                )
                if signal_pdf is None:
                    if isinstance(_output, tuple):
                        signal_pdf, normalization_factor = _output
                    else:
                        signal_pdf = _output
                else:
                    if isinstance(_output, tuple):
                        signal_pdf += _output[0]
                        normalization_factor.append(_output[1])
                    else:
                        signal_pdf += _output
                if not USE_NPRED_CACHE:
                    evaluator.reset_cache_properties()
        output = Map.from_geom(
            self._geom, data=signal_pdf.data / n, unit=signal_pdf.unit
        )
        if return_normalization_factor:
            output = (output, normalization_factor)
        return output

    def background_pdf(
        self, energy_min=None, energy_max=None, return_normalization_factor=False
    ):
        """Compute background probability of the model over the map energy range.

        Returns
        -------
        prob : `~numpy.ndarray`
            Background probability of the model.
        """
        energy_reco_axis = self.edisp_kernel_unbinned.axes[
            "energy"
        ]  # extremely important to stay consistent with the normalization computed for signal_pdf, the integration is done between energy_reco limits that are the same as the ones used in the
        data = self.background_model(energy_reco_axis.center)
        flux = Map.from_geom(geom=self.geom, data=data.value, unit=data.unit)
        # NB no need for bkg bias because it's normalized
        normalization_factor = self.background_model.integral(
            energy_min=energy_reco_axis.center.min()
            if energy_min is None
            else energy_min,
            energy_max=energy_reco_axis.center.max()
            if energy_max is None
            else energy_max,
        )
        pdf = flux / normalization_factor
        # pdf.data = np.clip(pdf.data, 0, None) #to avoid negative pdf values
        if return_normalization_factor:
            return pdf, normalization_factor
        else:
            return pdf

    def set_background_norm_prior(self):
        log.debug("applying prior on the background model based on safe mask")
        mu = (
            len(self.events_off_safe.energy)
            * np.mean(self.alpha.data)
            / self.background_model.integral(
                energy_min=self.events_off_safe.energy.min(),
                energy_max=self.events_off_safe.energy.max(),
            )
            * self.background_model.norm.value
        )
        # if hasattr(dataset, "bias_background"):
        #    dataset.background_model.norm.frozen = True
        #    dataset.background_model.norm.value = mu
        #    dataset.background_model.error = np.abs(1 - mu) / 2
        # else :
        self.background_model.norm.prior = GaussianPrior(mu=mu, sigma=np.abs(1 - mu))
        self.background_model.norm.value = mu

    @property
    def mask_safe(self):
        """Getter for mask_safe"""
        return super().mask_safe

    @mask_safe.setter
    def mask_safe(self, value):
        """Setter for mask_safe with custom processing"""
        self._mask_safe = value
        if hasattr(self, "background_model") and self.background_model is not None:
            self.set_background_norm_prior()

    @property
    def events_off_safe(self):
        mask = np.logical_and(
            self.events_off.energy >= self.events_safe.energy.min(),
            self.events_off.energy <= self.events_safe.energy.max(),
        )
        return self.events_off.select_row_subset(mask)

    def plot_results(self, figsize=(10, 6)):
        """Plot the results of the fit."""
        names = []
        evaluators = []
        for item in self.evaluators.items():
            names.append(item[0])
            evaluators.append(item[1])
        ncols = len(names)
        fig, axs = plt.subplots(ncols, 1, figsize=figsize)
        axs = np.asarray(axs).flatten()
        for i, (name, _evaluator) in enumerate(zip(names, evaluators)):
            ax = axs[i]
            ax.set_title(f"Model : {name}")
            flux = _evaluator.compute_flux()
            flux = _evaluator.apply_exposure(flux)
            flux = _evaluator.apply_edisp(flux, edisp=_evaluator.edisp_e_reco_binned)

            flux.data = flux.data / flux.geom.axes["energy"].bin_width.value.reshape(
                -1, 1, 1
            )
            ax = flux.plot_hist(ax=ax, label="Unbinned fit")

            hist, _ = np.histogram(
                self.events.energy, bins=flux.geom.axes["energy"].edges
            )
            ax.plot(
                flux.geom.axes["energy"].center.value,
                hist / flux.geom.axes["energy"].bin_width,
                lw=2,
                label="On events for unbinned",
                drawstyle="steps-mid",
            )

            hist, _ = np.histogram(
                self.events_off.energy,
                bins=flux.geom.axes["energy"].edges,
                weights=self.alpha.data.flatten() if self.alpha is not None else None,
            )
            ax.plot(
                flux.geom.axes["energy"].center.value,
                hist / flux.geom.axes["energy"].bin_width,
                drawstyle="steps-mid",
                lw=2,
                label="Off events for unbinned",
            )

            ax.plot(
                flux.geom.axes["energy"].center.value,
                1
                / self.background_model.norm.value
                * self.background_model(flux.geom.axes["energy"].center),
                label="bkg model initial for unbinned spline",
                ls="--",
            )

            ax.plot(
                flux.geom.axes["energy"].center.value,
                self.background_model(flux.geom.axes["energy"].center),
                label="bkg model fitted for unbinned spline",
                ls="-.",
            )

            # ax.set_xlim(0.01, 1e2)
            # ax.set_ylim(1e-2, 1e6)
            ax.legend()
        fig.tight_layout()
        return fig, ax
