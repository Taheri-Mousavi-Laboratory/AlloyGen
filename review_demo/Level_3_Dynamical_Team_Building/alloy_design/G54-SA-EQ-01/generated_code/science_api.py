from pathlib import Path
import os, sys
REPO = Path(os.environ['ALLOYGEN_REPOSITORY'])
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / 'experiments/management_alloy_design'))
from scientific_tools import AlloyDesignTools
OUTPUT = Path(os.environ['ALLOYGEN_SINGLE_OUTPUT'])
tools = AlloyDesignTools(OUTPUT / 'search')
