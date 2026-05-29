export LD_LIBRARY_PATH=$(find tf -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)
