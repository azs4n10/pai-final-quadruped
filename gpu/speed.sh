#!/bin/bash
cd /mnt/c/Users/mpmkk/Documents/apps/physical_ai/pai_final
for n in "$@"; do XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 timeout 900 ~/mjx/bin/python gpu/speed.py $n 2>&1 | grep steps; done
