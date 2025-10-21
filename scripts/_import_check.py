import sys
import pathlib
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

import amp.app
import amp.virtual_joystick

print('Imports OK')
