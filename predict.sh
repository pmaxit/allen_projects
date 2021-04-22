export OUTPUT_FILE=predictions.json

allennlp predict \
  --output-file $OUTPUT_FILE \
  --include-package my_project \
  --predictor conll_03_predictor \
  --use-dataset-reader \
  --silent \
  /tmp/test/4 \
  data/validation.txt
