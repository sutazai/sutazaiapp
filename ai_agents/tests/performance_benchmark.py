import os
import statistics
import tempfile
import time
from typing import Any, Dict

import cv2
import fitz  # type: ignore
import numpy as np
import pytest

from ai_agents.document_processor.src import DocumentProcessorAgent
from ai_agents.document_processor.utils.document_utils import DocumentUtils

# ... existing code ...
