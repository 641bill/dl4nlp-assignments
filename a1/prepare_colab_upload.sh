#!/usr/bin/env bash
# Create a zip bundle for uploading to Google Colab.
set -euo pipefail

cd "$(dirname "$0")"

ZIP_NAME="a1_colab_upload.zip"

echo "Creating ${ZIP_NAME} ..."

zip -r "${ZIP_NAME}" \
  A1_skeleton.py \
  part4.py \
  evaluate_model.py \
  train.txt \
  val.txt

echo ""
echo "Done. Upload this file to Colab:"
echo "  $(pwd)/${ZIP_NAME}"
echo ""
echo "Then open colab_train_50k.ipynb in Google Colab."
