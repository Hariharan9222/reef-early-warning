"""
src/bi/export_for_powerbi.py
Wrapper redirecting to powerbi/export_for_powerbi.py
"""
import os
import sys

# Ensure root in sys.path
sys.path.append(os.path.abspath("."))
from powerbi.export_for_powerbi import main

if __name__ == "__main__":
    main()

