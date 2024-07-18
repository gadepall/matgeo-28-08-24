#!/bin/bash

# Find all LaTeX files recursively in the current directory and its subdirectories
find . -type f -name "*.tex" | while IFS= read -r file; do
    echo "Processing file: $file"

    # Use sed to modify figure environments
    # Change \begin{figure}[...]
    sed -i 's/\\begin{figure}\[.*\]/\\begin{figure}[H]/g' "$file"
    sed -i 's/\\begin{table}\[.*\]/\\begin{table}[H]/g' "$file"

    # Change \includegraphics[...]{...}
    sed -i 's/\\includegraphics\[\(.*\)\]{\([^}]*\)}/\\includegraphics\[width=0.75\\columnwidth\]{\2}/g' "$file"
done

echo "Figure settings modified in all *.tex files."

