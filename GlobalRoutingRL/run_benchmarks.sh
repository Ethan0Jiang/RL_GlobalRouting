#!/bin/bash
# Bash script to run the Python program with different benchmarks

# Define the script to run and the base file paths
PYTHON_SCRIPT="main.py"
OUTPUT_DIR="output"
BENCHMARK_DIR="benchmark"

# Ensure the output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Loop from 1 to 5 and run the Python script with the corresponding files
for i in {1..5}; do
  INPUT_FILE="$BENCHMARK_DIR/test_benchmark_${i}.gr"
  OUTPUT_FILE="$OUTPUT_DIR/bm${i}_result.txt"
  
  # Run the Python script with the input and output files
  echo "Running benchmark ${i}..."
  python $PYTHON_SCRIPT $INPUT_FILE $OUTPUT_FILE
  
  # Check if the script ran successfully
  if [ $? -ne 0 ]; then
    echo "Error running benchmark ${i}. Exiting."
    exit 1
  fi
done

echo "All benchmarks ran successfully."

