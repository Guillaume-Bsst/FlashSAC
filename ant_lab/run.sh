#!/bin/bash
# usage: ant_lab/run.sh <args python...>  (ex: ant_lab/run.sh -m ant_lab.probe_model)
set -e
PY=$HOME/.holosoma_deps/miniconda3/envs/hssim/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$(dirname "$0")/.."
exec "$PY" "$@"
