import os
import xml.etree.ElementTree as ET
from PIL import Image

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def parse_points(points_str):
    pts = [tuple(map(int, p.split(","))) for p in points_str.strip().split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def crop_lines(image_dir, page_dir, out_dir, split_name, padding=8):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    counter = 0
    xml_files = sorted([f for f in os.listdir(page_dir) if f.endswith(".xml")])
    print(f"Processing {len(xml_files)} pages for: {split_name}", flush=True)
    for xml_file in xml_files:
        page_name = xml_file.replace(".xml", "")
        img_path = os.path.join(image_dir, page_name + ".JPG")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, page_name + ".jpg")
        if not os.path.exists(img_path):
            continue
        try:
            tree = ET.parse(os.path.join(page_dir, xml_file))
            root = tree.getroot()
            img = Image.open(img_path)
            W, H = img.size
            for region in root.findall(f".//{{{NS}}}TextRegion"):
                for line in region.findall(f"{{{NS}}}TextLine"):
                    coords_el = line.find(f"{{{NS}}}Coords")
                    text_el = line.find(f"{{{NS}}}TextEquiv/{{{NS}}}Unicode")
                    if coords_el is None or text_el is None:
                        continue
                    text = (text_el.text or "").strip()
                    if not text:
                        continue
                    x1, y1, x2, y2 = parse_points(coords_el.attrib["points"])
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(W, x2 + padding)
                    y2 = min(H, y2 + padding)
                    if x2 - x1 < 10 or y2 - y1 < 5:
                        continue
                    crop = img.crop((x1, y1, x2, y2))
                    out_name = f"{split_name}_{counter}.jpeg"
                    crop.save(os.path.join(out_dir, out_name), "JPEG")
                    entries.append(f"{out_name}\t{text}")
                    counter += 1
        except Exception as e:
            print(f"  ERROR {xml_file}: {e}", flush=True)
    print(f"Done {split_name}: {len(entries)} lines", flush=True)
    return entries

OUT  = r"C:\Users\anjal\Downloads\HTR-ConvText\data\read2016\lines"
TEST_BASE = r"C:\Users\anjal\Downloads\READ2016_Test\Test-ICFHR-2016"
VAL_BASE  = r"C:\Users\anjal\Downloads\READ2016\PublicData\Validation"

print("=== VALIDATION ===", flush=True)
val = crop_lines(
    image_dir  = os.path.join(VAL_BASE, "Images"),
    page_dir   = os.path.join(VAL_BASE, "page"),
    out_dir    = OUT,
    split_name = "val"
)

print("=== TEST ===", flush=True)
test = crop_lines(
    image_dir  = TEST_BASE,
    page_dir   = os.path.join(TEST_BASE, "page"),
    out_dir    = OUT,
    split_name = "test"
)

print(f"\nFINAL: val={len(val)} test={len(test)}", flush=True)
