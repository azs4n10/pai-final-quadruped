#!/bin/bash
cd /mnt/c/Users/mpmkk/Documents/apps/physical_ai/pai_final
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 timeout 900 ~/mjx/bin/python gpu/speed_go1.py Go1JoystickRoughTerrain 128 2>&1 | grep -E "steps|Error" | tail -2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 timeout 900 ~/mjx/bin/python gpu/speed_go1.py Go1JoystickRoughTerrain 1024 2>&1 | grep -E "steps|Error" | tail -2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 timeout 900 ~/mjx/bin/python gpu/speed_go1.py Go1JoystickFlatTerrain 1024 2>&1 | grep -E "steps|Error" | tail -2
