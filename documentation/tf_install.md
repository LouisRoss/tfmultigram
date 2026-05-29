sudo apt install "tensorflow[and-cuda]"
export LD_LIBRARY_PATH=$(find tf -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
