exoTEDRF v2 (JAX)
=================

``exotedrf.v2`` is an optional, memory-resident implementation of the exoTEDRF
reduction and optimizer written in `JAX <https://jax.readthedocs.io>`_. It
supports NIRISS/SOSS, NIRSpec/BOTS and MIRI/LRS time-series observations,
reads the same ``run_optimize.yaml`` and ``run_DMS.yaml`` configuration files
as the original pipeline, and produces the same Stage-2/Stage-3 products.

Instead of writing every intermediate step to disk and re-reading it, v2
keeps the observation in accelerator or host memory, runs each calibration
step as a compiled (jitted) kernel, and re-runs only the part of the step graph
downstream of the parameter being optimized. The original pipeline
(``exotedrf.stage1`` ... ``exotedrf.optimize``) is unchanged and remains the
reference implementation; v2 is validated against it.

Installation
------------

v2 follows exoTEDRF 2.5.0 and is tested with Python 3.13 and jwst 3.0.0.
It is an optional extra. For a CPU-only installation:

.. code-block:: bash

   pip install -e ".[v2-cpu]"

For a GPU, first install a CUDA build of JAX that matches your driver (see the
JAX installation guide), then install exoTEDRF without re-resolving its
dependencies and confirm the backend:

.. code-block:: bash

   pip install --no-deps -e .
   python -c "import jax; print(jax.default_backend(), jax.devices())"

NVIDIA A100 and V100 GPUs have been tested. Set a persistent compilation cache
so that kernels are compiled only once per machine:

.. code-block:: bash

   export EXOTEDRF_JAX_CACHE=/path/to/jax-cache

Running
-------

The optimizer takes the same configuration file as
``exotedrf/optimize.py``:

.. code-block:: bash

   python -m exotedrf.v2.optimize --config run_optimize.yaml

and the single-pass pipeline (no optimization) takes a ``run_DMS.yaml``:

.. code-block:: bash

   python -m exotedrf.v2.run_dms --config run_DMS.yaml

Outputs are written below ``<pipeline_outputs_directory>[_<output_tag>]/v2``
(override with ``--output-dir``): the v1-style ``Cost``/``Scatter`` logs, a JSON
optimizer summary, the final ``Stage2/<name>_rateints_v2.fits`` cube, the
``Stage3/<target>_<method>_spectra_fullres.fits`` spectrum, the reusable
CSV/NPY sidecars (centroids, background, bad-pixel map, PCA stability, light
curve estimate) and the v1 diagnostic plots. Stage-3 spectra include a DQ
report for each order: 1 for DO_NOT_USE, 2 for saturation, 4 for hot/warm
pixels and 8 for high variance. The report uses integration 10 (zero-based),
or the last integration for shorter visits; ATOCA reports -1 where this
diagnostic is unavailable. Per-step intermediate FITS files are not written.

Useful optimizer options:

``--output-mode optimal``
   Write only the final spectrum, the optimizer JSON and bounded diagnostic
   plots (skips the rate cube and sidecars). Same calculation.
``--no-products``
   Skip all product writing, e.g. for timing runs.
``--x64``
   Run in float64 (validation only; production uses float32).
``--search greedy|joint|beam|tree``
   Phase-1 search strategy (see below). ``greedy`` reproduces v1.

Special run modes
~~~~~~~~~~~~~~~~~

The v1 run modes are available with the same YAML keys:

- ``optimize_extract_width_only``: re-optimize only the extraction width from
  the Stage-2 products of a previous run.
- ``from_pca_only`` / ``optimize_from_pca_only``: restart from existing Stage-2
  products just before PCA reconstruction.
- ``reuse_first_pass_extract_width`` with ``first_pass_extract_method``: take
  the extraction width from a previous run's log or Stage-3 header.
- ``input_filetag``: restart from an intermediate v1 product (for example
  ``gainscalestep``, ``rateints`` or ``badpixstep``) instead of ``uncal``.
- ``debug_mode`` and ``archive_to_longterm_storage``.

Supported processing options
----------------------------

v2 supports the exoTEDRF 2.5.0 reduction options, including SOSS
``scale-achromatic``, ``scale-achromatic-window``, ``scale-chromatic`` and
``solve`` 1/f correction, NIRSpec ``median`` and ``slope`` correction,
up-the-ramp and time-domain jump detection, symmetric or asymmetric box
apertures, NIRSpec/MIRI optimal extraction, SOSS ATOCA extraction, and
stellar-model or PASTASOSS wavelength refinement.

Extraction defaults to ``mask_do_not_use_pixels: False`` and
``mask_saturated_pixels: False``. Set either option to ``True`` to exclude
those pixels. For NIRISS, ``saturation_rescue: True`` overrides the saturated
pixel mask. Bad-pixel correction defaults to ``median_high_variance: False``,
``preserve_saturated: False`` and ``clear_interpolated_dq: False``.

Stage-3 light curves use a clipping threshold of 10 and a window of 10.
Override the threshold with
``stage3_kwargs: {Extract1dStep: {clip_thresh: 10}}``.

ATOCA is a wrapper around the installed JWST pipeline, called on in-memory
datamodels and parallelized over segments and integration chunks. ATOCA
results depend on the BLAS thread count (in v1 as well); worker processes
inherit the parent's setting.

Configuration compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Settings that v1 accepts but ignores or silently changes are handled the same
way in v2, with a warning: for example a NIRSpec ``oof_method:
scale-achromatic`` becomes ``median``, a NIRISS custom superbias becomes
``crds``, and steps that v1 does not run for an instrument are skipped.
``ResetStep``, ``Extract2DStep``, ``SourceTypeStep`` and ``WaveCorrStep`` were
removed in 2.5.0; their old run switches and keywords are accepted and
ignored with a warning.
``stage1_kwargs``/``stage2_kwargs``/``stage3_kwargs`` entries are translated to
their v2 equivalents; a keyword without a faithful v2 equivalent raises an
error instead of being ignored.

v2-specific options
~~~~~~~~~~~~~~~~~~~

All are optional; the defaults reproduce v1 behavior.

- ``v2_output_mode``: ``standard`` (default) or ``optimal`` (see above).
- ``v2_search_strategy``: ``greedy`` (default, v1 coordinate descent),
  ``joint`` (full grid per checkpoint group), ``beam`` (keep the best
  ``v2_beam_width`` configurations per group and score the best
  ``v2_final_candidates`` on the full data) or ``tree`` (adaptive beam that
  keeps candidates within ``v2_tree_tolerance`` of the best). Large groups fall
  back to iterated coordinate descent above ``v2_group_max_evals`` grid points.
- ``v2_max_host_bytes`` and ``v2_scratch_dir``: limit the arrays held in host
  memory and choose where larger arrays are spilled.
- ``v2_stream_stage1``: ``auto`` (default), ``true`` or ``false``; processes
  Stage 1 one FITS segment at a time. Results are identical either way.
- ``v2_atoca_workers``, ``v2_atoca_solver_threads``, ``v2_atoca_chunk_ints``:
  ATOCA parallelism.
- ``v2_stellar_model_dir``: cache directory for downloaded PHOENIX stellar
  models (defaults to the run's Stage-3 directory).
- ``v2_upramp_cpu_count``: reproduce the row slicing of a v1 up-the-ramp run
  made on a machine with a different CPU count.
- ``refpack``: use a prebuilt reference pack (see below).

Reference packs and CRDS
------------------------

The JAX kernels do not call JWST pipeline steps. Instead, v2 extracts the
required CRDS reference arrays (superbias, linearity, mask, read noise, gain,
dark, flat, wavelength solution and instrument-specific references) once per
observation into a compact ``.npz`` reference pack. Building a pack needs a
working CRDS/JWST installation; loading one needs only NumPy and Astropy.

As in v1, the CRDS context is pinned by ``crds_context`` in the YAML (default
``jwst_1322.pmap``); a ``CRDS_CONTEXT`` environment variable takes precedence.
Reference files are downloaded once into ``crds_cache_path`` and reused. The
pack is cached as ``<output>/v2/refpacks/refpack_*_<context>.npz`` and reused
by later runs into the same output directory; a different context, detector,
subarray or wavelength map produces a new pack, and a pack whose recorded
context differs from the requested one is rejected. To use a newer CRDS
release, change ``crds_context`` explicitly.

SOSS uses the reference files distributed in the repository's ``files/``
directory: ``jwst_niriss_spectrace_0022.fits`` for SUBSTRIP96 and
``jwst_niriss_spectrace_0023.fits`` for SUBSTRIP256, the corresponding
``jwst_niriss_wavemap_0020.fits`` and ``jwst_niriss_wavemap_0022.fits``, and
``jwst_niriss_photom_rev2.fits``. The ``model_background96.npy`` and
``model_background256.npy`` files provide the matching backgrounds. Installed
packages also search ``sys.prefix/files``. Missing SOSS references can be
downloaded from the upstream repository.

A pack can also be built directly:

.. code-block:: bash

   python -m exotedrf.v2.refs first_segment_uncal.fits \
       --context jwst_1322.pmap --out refpack.npz

Precision and validation
------------------------

Production runs use float32. Ramp fitting uses a centered, segmented
closed-form least-squares solution, which keeps float32 covariances stable.
Float64 (``--x64``) must run in a separate process because JAX's precision
setting is process-wide.

The test suite (``tests/v2``) checks every kernel against a NumPy reference
transcribed from v1 or against the v1 functions themselves, and runs
end-to-end synthetic reductions for all three instruments:

.. code-block:: bash

   python -m pytest tests/v2

``python -m exotedrf.v2.validate`` compares checkpoint bundles from two runs
(v1 against v2, or float32 against float64) step by step with explicit
tolerances; DQ arrays always require exact equality.

Real-data comparisons with exoTEDRF 2.5.0 and jwst 3.0.0 give the following
results. Each row states the integration subset used; these measurements
are independent checks of the indicated calculation.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Calculation
     - Data subset
     - Agreement with v1
   * - SOSS jump detection and ramp DQ
     - 20 integrations
     - 65,126/65,126 JUMP_DET flags agree; no DQ or rate NaN-mask mismatches
   * - SOSS ramp rates
     - 20 integrations
     - Relative difference: p99 2.17e-7, maximum 2.51e-5; maximum absolute
       difference 1.53e-4 DN/s
   * - SOSS ramp errors
     - 20 integrations
     - Relative difference: p99 1.14e-7, maximum 1.94e-6
   * - SOSS reference-pixel correction
     - 10 integrations
     - Maximum absolute difference 0.001953125 DN
   * - SOSS ATOCA extraction
     - 5 integrations
     - Both orders' wavelength, flux and error arrays match exactly
   * - NIRSpec G395H NRS1 wavelengths
     - Real observation
     - Bit-identical; maximum difference 0 microns
   * - Streamed SOSS PCA
     - 537 integrations
     - Minimum absolute component cosine 0.999999999968 for components
       with variance ratio above 1e-6; reconstruction error 2.37e-6 of the data range

Performance
-----------

Runtime depends on the observation, search grid, hardware and compilation
cache. On synthetic float32 cubes with eight CPU cores, PCA fitting and
reconstruction took 22.4 s for 2000 x 256 x 2048 samples (exact solver) and
49.1 s for 16384 x 32 x 2048 samples (randomized solver). GPU timings for
those PCA sizes have not been validated.

Differences from v1
-------------------

v2 deliberately deviates from v1 in the following cases; each is reported in
the optimizer summary or as a warning.

- The optimizer commits the actual minimum-cost trial (v1 can commit the last
  trial), propagates winning structural window sizes to later sweeps, and
  never selects a non-finite cost.
- MIRI phase-1 jump scoring uses the newest usable group rather than a group
  that DQ initialization flags as ``DO_NOT_USE`` for every pixel.
- Group-level ``oof_method: solve`` works without ``outlier_maps``; v1 raises
  an ``IndexError`` there. v2 applies the reset-artifact mask instead.
- Integration-level ``even_odd_rows: False`` applies one level per column; the
  corresponding v1 code path fails.
- Up-the-ramp jump detection: the across-integration sigma-clipping branch of
  the JWST algorithm is not implemented (v1's settings never reach it), and
  ``find_showers: True`` is rejected.
- ATOCA combined with stellar wavelength refinement or ``use_pastasoss`` is
  not yet supported and raises an error, as does ATOCA for an F277W exposure.
- Per-step intermediate FITS files are not written.

On CPU, MIRI up-the-ramp jump detection is currently about two times slower
than the JWST implementation; all other options are faster.
