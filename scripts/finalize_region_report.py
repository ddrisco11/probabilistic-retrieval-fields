"""Regenerate the site and charts from a saved experiment."""
import json
from pathlib import Path
from prf.region_report import report
from prf.storage import save_region_results

out=Path('results/wikiqa-regions');payload=json.loads((out/'results.json').read_text())
report(payload,out)
if 'per_query' in payload['results'][0]:save_region_results(payload,out)
Path('docs/index.html').write_text((out/'explorer.html').read_text())
Path('docs/.nojekyll').touch()
