"""Print the structure of the raw IDSSE files: tag and attribute names and counts, no values."""
import collections
import sys
import xml.etree.ElementTree as ET

from external.idsse import raw_path

match = sys.argv[1] if len(sys.argv) > 1 else "J03WPY"
for kind in ("meta", "event", "tracking"):
    tags = collections.Counter()
    attrs = collections.defaultdict(set)
    for _, el in ET.iterparse(raw_path(match, kind), events=("end",)):
        tags[el.tag] += 1
        attrs[el.tag].update(el.attrib)
        if el.tag in ("FrameSet", "Event"):
            el.clear()
    print("== {}: {} tags".format(kind, len(tags)))
    for tag, n in tags.most_common(25 if kind != "event" else 12):
        print("  {:<22} {:>9}  {}".format(tag, n, sorted(attrs[tag])[:22]))
