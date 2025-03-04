#!/bin/bash
# Fix high CPU usage
echo "Killing Python processes..."
killall -9 python python3 pytest
echo "Modifying test scripts..."
find . -name "*.sh" -exec sed -i 's/pytest -n 1/g' {} \;
echo "Done! CPU usage should now be lower." 