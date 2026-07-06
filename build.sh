#!/bin/bash
set -e
cd "$(dirname "$0")/src"
hipcc -O3 --offload-arch=gfx936 -shared -fPIC -o ../libdsv4_all_hip.so launchers.hip
