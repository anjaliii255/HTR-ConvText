import os

OUT_DIR = r"C:\Users\anjal\Downloads\HTR-ConvText\data\read2016\lines"

# Re-read the tab-separated entries we generated
# We need to re-run cropping to get labels, OR read from our saved data
# Lets read the .ln files + regenerate labels from XML

import xml.etree.ElementTree as ET
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def parse_points(points_str):
    pts = [tuple(map(int, p.split(","))) for p in points_str.strip().split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def generate_txts(page_dir, split_name, out_dir):
    counter = 0
    xml_files = sorted([f for f in os.listdir(page_dir) if f.endswith(".xml")])
    print(f"Generating txt files for {split_name} from {len(xml_files)} pages", flush=True)
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(page_dir, xml_file))
        root = tree.getroot()
        for region in root.findall(f".//{{{NS}}}TextRegion"):
            for line in region.findall(f"{{{NS}}}TextLine"):
                coords_el = line.find(f"{{{NS}}}Coords")
                text_el   = line.find(f"{{{NS}}}TextEquiv/{{{NS}}}Unicode")
                if coords_el is None or text_el is None:
                    continue
                text = (text_el.text or "").strip()
                if not text:
                    continue
                x1,y1,x2,y2 = parse_points(coords_el.attrib["points"])
                if x2-x1 < 10 or y2-y1 < 5:
                    continue
                txt_name = f"{split_name}_{counter}.txt"
                txt_path = os.path.join(out_dir, txt_name)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                counter += 1
    print(f"Done {split_name}: {counter} txt files", flush=True)
    return counter

BASE     = r"C:\Users\anjal\Downloads\READ2016\PublicData"
TEST_BASE= r"C:\Users\anjal\Downloads\READ2016_Test\Test-ICFHR-2016"
OUT      = r"C:\Users\anjal\Downloads\HTR-ConvText\data\read2016\lines"

print("=== Generating label txt files ===", flush=True)
generate_txts(os.path.join(BASE,"Training","page"),   "train", OUT)
generate_txts(os.path.join(BASE,"Validation","page"),  "val",   OUT)
generate_txts(os.path.join(TEST_BASE,"page"),          "test",  OUT)
print("ALL DONE", flush=True)
