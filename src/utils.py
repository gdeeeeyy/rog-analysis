import os
import csv
import re

def parse_csv(csv_path):
    """Parses the agni-manual-logging CSV and returns a dictionary mapping ImageNo to Tag."""
    image_tags = {}
    if not csv_path or not os.path.exists(csv_path):
        return image_tags
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_no = row['ImageNo'].strip()
            tag = row['ImageTag'].strip()
            image_tags[img_no] = tag
    return image_tags

def ext_num(fname):
    """Extracts the integer number from a filename."""
    m = re.search(r'\d+', fname)
    return int(m.group()) if m else -1

def get_cycle_files(image_dir, csv_file, process_tags):
    """Groups image filenames into cycles based on their index and tags."""
    image_tags = parse_csv(csv_file)
    target_tags = [t.strip() for t in process_tags.split(',')]
    
    valid_ext = ('.png', '.jpg', '.tif', '.tiff', '.jpeg')
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found.")
        return {}
        
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
    
    cycle_files = {}
    for f in files:
        num = ext_num(f)
        tag = image_tags.get(str(num), 'Usable') # default usable if no csv
        if tag in target_tags or not csv_file:
            cycle = ((num - 1) // 250) + 1
            if cycle not in cycle_files: 
                cycle_files[cycle] = []
            cycle_files[cycle].append((num, f))
            
    return cycle_files
