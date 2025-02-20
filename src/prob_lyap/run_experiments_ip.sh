#!/bin/bash
python main.py --progress-bar --run-all-objectives -nh 32 -s 0 -t 200_000 --ckpt-every 10_000 --env-id InvertedPendulum-v4
cd eval
python vector_field_plot_2D.py --full-experiment

file_count=$(find "./data/2D" -type f -name 'obj_data_2D_*.csv' | sed -E 's/.*obj_data_2D_([0-9]+)\.csv/\1/' | sort -n | tail -1)
echo "File count: $file_count"
python lvf_over_train.py --lsac-file "obj_data_2D_$file_count" -b 999_99999
# echo "Largest image: $largest_image"
# largest_image=$(find "." -type f -name 'lvf_lsac_*.png' | sed -E 's/.*lvf_lsac_([0-9]+)\.png/\1/' | sort -n | tail -1)
# explorer.exe "lvf_lsac_$largest_image.png"


# Find the latest lvf_lsac file
echo "Searching for lvf_lsac files..."
largest_image=$(find "." -type f -name 'lvf_lsac_2D_*.png' | sed -E 's/.*lvf_lsac_2D([0-9]+)\.png/\1/' | sort -rn | head -1)
echo "Largest image: $largest_image"

if [ -z "$largest_image" ]; then
    echo "Error: No lvf_lsac files found."
    exit 1
fi

# Open the latest lvf_lsac file
output_file="lvf_lsac_2D_${largest_image}"
echo "Opening file: $output_file"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "$output_file"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open "$output_file"
else
    xdg-open "$output_file" || echo "Error: Unable to open $output_file"
fi
