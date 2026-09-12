"""Helpers for `local/REPORT_final_FINAL.ipynb`.

Submodules are imported explicitly rather than re-exported here, so that
`%autoreload 2` reliably picks up edits to any of them:

- `style`              typography, restyle layer, PDF/CSV saving
- `vaidya_targets`     frozen experimental reference values from Vaidya et al.
- `bioplausibility`    model-vs-CA1 statistics, overlay figures, scorecard
- `eval_errors`        hardcoded checkpoint eval errors by experiment
- `eval_error_plots`     supplementary eval-error bar charts
- `training_loss_plots`       two-rooms training loss (wraps existing plot helper)
- `single_room_ratemap_plots` final-segment ratemap peak signal and rank mosaics
- `single_room_pf_plots`      normalized revive metrics from PF tracking
- `two_rooms_pf_plots`        two-room alive proportion (wraps existing plot helper)
"""
