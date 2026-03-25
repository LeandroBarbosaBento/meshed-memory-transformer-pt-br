#!/usr/bin/env python3
"""
Convert Portuguese COCO captions from Parquet format to standard COCO JSON format.
"""

import pandas as pd
import json
import os

def convert_parquet_to_coco_format(parquet_dir, output_dir):
    """
    Convert parquet files to COCO JSON format.
    
    Expected parquet structure:
    - image: binary image data
    - caption: array of 5 captions
    - cocoid: COCO image ID
    - imgid: unique image ID
    - filename: image filename
    - split: train/validation/test
    """
    
    print(f"Reading parquet files from: {parquet_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all parquet files
    all_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    print(f"\nFound {len(all_files)} parquet files")
    
    dfs = []
    for file in sorted(all_files):
        file_path = os.path.join(parquet_dir, file)
        df_temp = pd.read_parquet(file_path)
        print(f"  Loaded {file}: {len(df_temp)} rows")
        dfs.append(df_temp)
    
    # Concatenate all DataFrames
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(df)} rows")
    print(f"Splits: {df['split'].unique()}")
    
    # Process each split
    splits_map = {
        'train': 'captions_train2014.json',
        'val': 'captions_val2014.json', 
        'test': 'captions_test2014.json',
        'restval': 'captions_restval2014.json'
    }
    
    for split_name, output_file in splits_map.items():
        split_df = df[df['split'] == split_name]
        
        if len(split_df) == 0:
            print(f"\nSkipping {split_name} (no data)")
            continue
            
        print(f"\nProcessing {split_name}: {len(split_df)} images")
        
        # Build COCO format
        images = []
        annotations = []
        annotation_id = 1
        
        for idx, row in split_df.iterrows():
            image_id = int(row['imgid'])
            
            # Add image info
            images.append({
                'id': image_id,
                'file_name': row['filename'],
                'coco_id': int(row['cocoid'])
            })
            
            # Add captions (array of 5 captions per image)
            captions = row['caption']
            for caption_text in captions:
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'caption': caption_text
                })
                annotation_id += 1
        
        # Create COCO JSON structure
        coco_format = {
            'images': images,
            'annotations': annotations,
            'type': 'captions',
            'info': {
                'description': f'COCO {split_name} - Portuguese Captions',
                'version': '1.0',
                'year': 2024
            }
        }
        
        # Save to file
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, ensure_ascii=False, indent=2)
        
        print(f"  → Saved {len(images)} images, {len(annotations)} captions to {output_file}")

if __name__ == '__main__':
    # Directories
    parquet_dir = '/home/leandro/tcc/meshed-memory-transformer/coco-captions-pt-br/data'
    output_dir = '/home/leandro/tcc/meshed-memory-transformer/annotations'
    
    # Convert
    convert_parquet_to_coco_format(parquet_dir, output_dir)
    
    print("\n=== Conversion complete! ===")
