#!/bin/bash
# Clean up the system
apt-get autoremove -y
apt-get autoclean -y
echo "System cleanup completed successfully!" 