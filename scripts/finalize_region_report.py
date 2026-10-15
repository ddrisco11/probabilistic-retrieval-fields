"""Regenerate the site and charts from a saved experiment."""
import json
from pathlib import Path
from prf.region_report import report
from prf.storage import save_region_results

out=Path('results/wikiqa-regions');payload=json.loads((out/'results.json').read_text())
report(payload,out)
if 'per_query' in payload['results'][0]:save_region_results(payload,out)
# Use granularity_report.py to combine follow-up runs into the public site.
if not Path('results/wikiqa-1000-regions/results.json').exists():
    Path('docs/index.html').write_text((out/'explorer.html').read_text())
Path('docs/.nojekyll').touch()
