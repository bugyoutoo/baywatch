import os
import xml.etree.ElementTree as ET
import pandas as pd

# Folder containing images and XML files
folder = "data/train"

# List all XML files
xml_files = [f for f in os.listdir(folder) if f.endswith('.xml')]

data = []

for xml_file in xml_files:
    xml_path = os.path.join(folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract filename from XML
    filename = root.find('filename').text

    # Initialize labels (adjust based on your XML content)
    debris = 0
    cloud = 0

    # Example: Suppose labels are stored as object names under <object><name> tags
    # You can adjust this part according to your XML structure
    
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if 'debris' in name:
            debris = 1
        if 'cloud' in name:
            cloud = 1

    data.append({
        'filename': filename,
        'debris': debris,
        'cloud': cloud
    })

# Create DataFrame and save CSV
df = pd.DataFrame(data)
df.to_csv('data/labels_parsed.csv', index=False)

print(f"Parsed {len(data)} XML files and saved CSV as 'data/labels_parsed.csv'")