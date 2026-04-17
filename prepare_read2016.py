import os
import xml.etree.ElementTree as ET
from PIL import Image

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def parse_points(points_str):
    pts = [tuple(map(int, p.split(','))) for p in points_str.strip().split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def crop_lines(image_dir, page_dir, out_dir, split_name, padding=8):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Dir created: {os.path.exists(out_dir)}")
    
    entries = []
    counter = 0
    xml_files = sorted([f for f in os.listdir(page_dir) if f.endswith('.xml')])
    print(f"Processing {len(xml_files)} XML files for split: {split_name}")

    for xml_file in xml_files:  # test with 5 pages first
        print(f"  Processing {xml_file}...")
        page_name = xml_file.replace('.xml', '')
        img_path = os.path.join(image_dir, page_name + '.JPG')
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, page_name + '.jpg')
        if not os.path.exists(img_path):
            print(f"    MISSING: {img_path}")
            continue

        try:
            tree = ET.parse(os.path.join(page_dir, xml_file))
            root = tree.getroot()
            img = Image.open(img_path)
            W, H = img.size
            print(f"    Image size: {W}x{H}")

            line_count = 0
            for region in root.findall(f'.//{{{NS}}}TextRegion'):
                for line in region.findall(f'{{{NS}}}TextLine'):
                    coords_el = line.find(f'{{{NS}}}Coords')
                    text_el   = line.find(f'{{{NS}}}TextEquiv/{{{NS}}}Unicode')
                    if coords_el is None or text_el is None:
                        continue
                    text = (text_el.text or '').strip()
                    if not text:
                        continue
                    x1, y1, x2, y2 = parse_points(coords_el.attrib['points'])
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(W, x2 + padding)
                    y2 = min(H, y2 + padding)
                    if x2 - x1 < 10 or y2 - y1 < 5:
                        continue
                    crop = img.crop((x1, y1, x2, y2))
                    out_name = f"{split_name}_{counter}.jpeg"
                    out_path = os.path.join(out_dir, out_name)
                    crop.save(out_path, "JPEG")
                    entries.append(f"{out_name}\t{text}")
                    counter += 1
                    line_count += 1
            print(f"    Cropped {line_count} lines")
        except Exception as e:
            print(f"    ERROR in {xml_file}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Total entries: {len(entries)}")
    if entries:
        print("Sample entries:")
        for e in entries[:3]:
            print(" ", e)
    return entries

if __name__ == "__main__":
    BASE    = r"C:/Users/anjal/Downloads/READ2016/PublicData"
    OUT_DIR = r"C:/Users/anjal/Downloads/HTR-ConvText/data/read2016/lines"

    print("=== TRAINING ===")
    train_entries = crop_lines(
        image_dir  = os.path.join(BASE, "Training", "Images"),
        page_dir   = os.path.join(BASE, "Training", "page"),
        out_dir    = OUT_DIR,
        split_name = "train"
    )