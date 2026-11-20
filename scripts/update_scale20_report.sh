#!/usr/bin/env bash
# Run independent verification before publishing a new report snapshot.
set -euo pipefail
REPORT_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPORT_CAMPAIGN=${1:?Supply the campaign directory}
REPORT_OUTPUT=${2:?Supply the report output directory}
source "$REPORT_REPO/scripts/jupiter_env.sh"
RFD_ANALYSIS_PYTHON="$LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv/bin/python"
"$RFD_ANALYSIS_PYTHON" "$REPORT_REPO/scripts/audit_scale20_campaign.py" "$REPORT_CAMPAIGN" --verify-models
"$RFD_ANALYSIS_PYTHON" "$REPORT_REPO/scripts/plot_scale20_telemetry.py" "$REPORT_CAMPAIGN" "$REPORT_OUTPUT"
"$RFD_ANALYSIS_PYTHON" "$REPORT_REPO/scripts/build_scale20_report.py" "$REPORT_CAMPAIGN" "$REPORT_OUTPUT"
