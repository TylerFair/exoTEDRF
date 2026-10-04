"""Apply the reference NIRISS reduction settings to test configurations."""

CURRENT_NIRISS_OVERRIDES = {
    'outlier_maps': None,
    'f277w': 'acceptance_f277w.npy',
    'centroids': None,
    'saturation_fraction': 0.8,
    'propagate_saturation': True,
    'saturation_threshold': None,
    'saturation_rescue': True,
    'mask_do_not_use_pixels': True,
    'baseline_ints': [50, -50],
    'pca_components': 10,
    'remove_components': None,
    'generate_lc': True,
    'flag_up_ramp': False,
    'flag_in_time': True,
}


def apply_current_niriss_settings(config):
    """Return a copy of the configuration with the reference SOSS settings.

    Parameters
    ----------
    config : dict
        Reduction configuration.

    Returns
    -------
    config : dict
        Configuration with reference SOSS settings.
    """
    result = dict(config)
    # Use the default inverse-linearity setting.
    result.pop('INLCorrStep', None)
    result.update(CURRENT_NIRISS_OVERRIDES)
    return result
