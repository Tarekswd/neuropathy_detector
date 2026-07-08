import xml.etree.ElementTree as ET
import os
import numpy as np
from pathlib import Path

# -------------------------------
# 1. Setup Paths and Output Directory
# -------------------------------
script_dir = Path(__file__).parent
xml_dir = script_dir / "XML_files_neuropathy" / "XML_files"
output_dir = script_dir / "fixed_pics"
output_dir.mkdir(exist_ok=True)

# Find all XML files in the folder
xml_files = sorted(list(xml_dir.glob("*.xml")))
print(f"Found {len(xml_files)} XML files to process.")

# Handle XML namespaces
ns = {'z': 'http://www.zebris.de/measurements'}
ET.register_namespace('', 'http://www.zebris.de/measurements')
ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

# Helper: parse space‑separated cells into a 2D numpy array
def parse_cells(text, nx, ny):
    numbers = list(map(float, text.strip().split()))
    return np.array(numbers).reshape((ny, nx))

# Helper: format 2D numpy array back to space-separated string
def format_cells(matrix):
    # Format with 1 decimal place to keep it clean and match original XML style
    flat = matrix.flatten()
    return " ".join(f"{x:.1f}" for x in flat)

# -------------------------------
# 2. Loop through all XML files
# -------------------------------
for file_idx, xml_path in enumerate(xml_files, start=1):
    file_name = xml_path.name
    print(f"[{file_idx}/{len(xml_files)}] Processing: {file_name}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  Error parsing XML {file_name}: {e}")
        continue
        
    # Get plate width W_plate from root cell_count
    cell_count_elem = root.find('z:cell_count', ns)
    if cell_count_elem is None:
        print(f"  Warning: No global cell_count found in {file_name}, skipping.")
        continue
        
    W_plate = int(cell_count_elem.findtext('z:x', namespaces=ns))
    
    # Process movements
    movements = root.findall('z:movements/z:movement', ns)
    modified = False
    
    for movement in movements:
        clips = movement.findall('z:clips/z:clip', ns)
        for clip in clips:
            events = clip.findall('z:data/z:event', ns)
            for event in events:
                side_elem = event.find('z:side', ns)
                if side_elem is not None and side_elem.text == 'left':
                    modified = True
                    
                    # 1. Change side to right
                    side_elem.text = 'right'
                    
                    # 2. Flip heel x coordinate
                    heel = event.find('z:heel', ns)
                    if heel is not None:
                        heel_x_elem = heel.find('z:x', ns)
                        if heel_x_elem is not None:
                            heel_x = float(heel_x_elem.text)
                            heel_x_elem.text = f"{W_plate - heel_x:.3f}"
                            
                    # 3. Flip toe x coordinate
                    toe = event.find('z:toe', ns)
                    if toe is not None:
                        toe_x_elem = toe.find('z:x', ns)
                        if toe_x_elem is not None:
                            toe_x = float(toe_x_elem.text)
                            toe_x_elem.text = f"{W_plate - toe_x:.3f}"
                            
                    # 4. Flip max pressure map
                    max_elem = event.find('z:max', ns)
                    if max_elem is not None:
                        cell_begin = max_elem.find('z:cell_begin', ns)
                        cell_count = max_elem.find('z:cell_count', ns)
                        cells_elem = max_elem.find('z:cells', ns)
                        
                        if cell_begin is not None and cell_count is not None and cells_elem is not None:
                            start_x_elem = cell_begin.find('z:x', ns)
                            nx_max = int(cell_count.findtext('z:x', namespaces=ns))
                            ny_max = int(cell_count.findtext('z:y', namespaces=ns))
                            
                            if start_x_elem is not None:
                                start_x = int(start_x_elem.text)
                                new_start_x = W_plate - (start_x + nx_max)
                                start_x_elem.text = str(new_start_x)
                                
                            max_matrix = parse_cells(cells_elem.text, nx_max, ny_max)
                            flipped_max = np.fliplr(max_matrix)
                            cells_elem.text = format_cells(flipped_max)
                            
                    # 5. Flip rollover sequence
                    rollover = event.find('z:rollover', ns)
                    if rollover is not None:
                        rollover_cell_count = rollover.find('z:cell_count', ns)
                        if rollover_cell_count is not None:
                            W_rollover = int(rollover_cell_count.findtext('z:x', namespaces=ns))
                            quants = rollover.findall('z:data/z:quant', ns)
                            for quant in quants:
                                q_cell_begin = quant.find('z:cell_begin', ns)
                                q_cell_count = quant.find('z:cell_count', ns)
                                q_cells_elem = quant.find('z:cells', ns)
                                
                                if q_cell_begin is not None and q_cell_count is not None and q_cells_elem is not None:
                                    q_start_x_elem = q_cell_begin.find('z:x', ns)
                                    q_nx = int(q_cell_count.findtext('z:x', namespaces=ns))
                                    q_ny = int(q_cell_count.findtext('z:y', namespaces=ns))
                                    
                                    if q_start_x_elem is not None:
                                        q_start_x = int(q_start_x_elem.text)
                                        q_new_start_x = W_rollover - (q_start_x + q_nx)
                                        q_start_x_elem.text = str(q_new_start_x)
                                        
                                    q_matrix = parse_cells(q_cells_elem.text, q_nx, q_ny)
                                    flipped_q = np.fliplr(q_matrix)
                                    q_cells_elem.text = format_cells(flipped_q)
                                    
    # Save output file (whether modified or not, to keep a complete dataset)
    output_path = output_dir / file_name
    try:
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        if modified:
            print(f"  Saved modified XML to: {output_path.name}")
        else:
            print(f"  Saved unmodified XML (already right side) to: {output_path.name}")
    except Exception as e:
        print(f"  Error writing XML {file_name}: {e}")

print("\nAll XML files processed successfully! Fixed XML data is saved in the 'fixed_pics' directory.")  explain the code