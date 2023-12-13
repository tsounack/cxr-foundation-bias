
#!/bin/bash

# Function to run disease prediction
run_prediction() {
  extractor=$1
  mode=$2
  n_runs=$3

  echo "Running disease prediction for $extractor extractor with $mode mode..."

  python disease-prediction.py --extractor $extractor --mode $mode --n_runs $n_runs

  echo "Disease prediction for $extractor extractor with $mode mode completed."
}

# Run disease prediction for foundation model
run_prediction "foundation" "baseline" 10

# Run disease prediction for densenet model
run_prediction "densenet" "baseline" 10
