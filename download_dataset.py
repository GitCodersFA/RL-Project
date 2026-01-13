"""
Download DAGM 2007 Dataset
The dataset contains grayscale images with pixel-level ground-truth masks for defects.
Class10 is used as per the paper.
"""
import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def download_dagm_dataset():
    """
    Download DAGM 2007 dataset from available sources
    """
    data_dir = 'data'
    dagm_dir = os.path.join(data_dir, 'DAGM')
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(dagm_dir) and len(os.listdir(dagm_dir)) > 0:
        print("DAGM dataset already exists!")
        return dagm_dir
    
    print("=" * 60)
    print("DAGM 2007 Dataset Download")
    print("=" * 60)
    
    # DAGM dataset URLs (the dataset is publicly available)
    # Primary source from Kaggle or alternative mirrors
    urls = [
        # Kaggle dataset (requires manual download if this fails)
        "https://www.kaggle.com/api/v1/datasets/download/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection"
    ]
    
    zip_path = os.path.join(data_dir, 'dagm.zip')
    
    # Try downloading from available sources
    downloaded = False
    for url in urls:
        try:
            print(f"\nDownloading from: {url}")
            download_file(url, zip_path)
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    if downloaded and os.path.exists(zip_path):
        extract_zip(zip_path, data_dir)
        os.remove(zip_path)  # Clean up zip file
        print(f"\nDataset downloaded to: {dagm_dir}")
    else:
        print("\n" + "=" * 60)
        print("AUTOMATIC DOWNLOAD FAILED")
        print("=" * 60)
        print("\nPlease download the DAGM 2007 dataset manually:")
        print("1. Go to: https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection")
        print("2. Download and extract to: data/DAGM/")
        print("\nAlternatively, download from Zenodo:")
        print("https://zenodo.org/record/12750201")
        print("\nThe dataset structure should be:")
        print("data/DAGM/")
        print("├── Class1/")
        print("│   ├── Train/")
        print("│   └── Test/")
        print("├── Class2/")
        print("...") 
        print("└── Class10/  <-- We use this class")
        
        # Create a placeholder structure for testing
        create_synthetic_dataset(dagm_dir)
    
    return dagm_dir

def create_synthetic_dataset(dagm_dir):
    """
    Create synthetic dataset for testing if real dataset unavailable
    """
    import numpy as np
    from PIL import Image
    
    print("\nCreating synthetic dataset for testing...")
    
    class_dir = os.path.join(dagm_dir, 'Class10')
    
    for split in ['Train', 'Test']:
        img_dir = os.path.join(class_dir, split)
        label_dir = os.path.join(class_dir, split, 'Label')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        num_samples = 1150 if split == 'Train' else 1150
        
        print(f"Creating {num_samples} {split.lower()} samples...")
        
        for i in tqdm(range(num_samples)):
            # Create grayscale image (512x512)
            img = np.random.randint(100, 200, (512, 512), dtype=np.uint8)
            
            # Add texture pattern
            x = np.linspace(0, 4*np.pi, 512)
            texture = np.outer(np.sin(x), np.cos(x)) * 20
            img = np.clip(img + texture, 0, 255).astype(np.uint8)
            
            # 50% have defects
            has_defect = i % 2 == 0
            
            # Create label mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            if has_defect:
                # Add synthetic defect (dark spot or scratch)
                cx, cy = np.random.randint(100, 412, 2)
                defect_type = np.random.choice(['spot', 'scratch'])
                
                if defect_type == 'spot':
                    # Circular defect
                    Y, X = np.ogrid[:512, :512]
                    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                    defect_mask = dist < np.random.randint(15, 40)
                    img[defect_mask] = img[defect_mask] - np.random.randint(30, 60)
                    mask[defect_mask] = 255
                else:
                    # Scratch defect
                    length = np.random.randint(50, 150)
                    angle = np.random.uniform(0, np.pi)
                    for t in range(length):
                        px = int(cx + t * np.cos(angle))
                        py = int(cy + t * np.sin(angle))
                        if 0 <= px < 512 and 0 <= py < 512:
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    if 0 <= px+dx < 512 and 0 <= py+dy < 512:
                                        img[py+dy, px+dx] = max(0, img[py+dy, px+dx] - 40)
                                        mask[py+dy, px+dx] = 255
            
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Save image
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(img_dir, f'{i:04d}.png'))
            
            # Save label mask
            if has_defect:
                mask_pil = Image.fromarray(mask)
                mask_pil.save(os.path.join(label_dir, f'{i:04d}_label.png'))
    
    # Create labels file
    for split in ['Train', 'Test']:
        labels_file = os.path.join(class_dir, split, 'labels.txt')
        with open(labels_file, 'w') as f:
            for i in range(1150):
                has_defect = 1 if i % 2 == 0 else 0
                if has_defect:
                    f.write(f'{i:04d}.png\t1\t{i:04d}_label.png\n')
                else:
                    f.write(f'{i:04d}.png\t0\n')
    
    print(f"\nSynthetic dataset created at: {dagm_dir}")
    print("Note: For best results, use the real DAGM 2007 dataset!")

if __name__ == '__main__':
    download_dagm_dataset()
