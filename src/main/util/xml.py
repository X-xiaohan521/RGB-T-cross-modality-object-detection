import xmltodict as xtd

def xml_to_dict(xml_path: str) -> dict:
    try:
        with open(xml_path, encoding="utf-8") as f:
            label_content = f.read()
            label_dict = xtd.parse(label_content)
            return label_dict
    except:
        raise RuntimeError(f"Error loading xml file: {xml_path}")