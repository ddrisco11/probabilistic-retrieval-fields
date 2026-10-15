"""Download the BEIR archive and verify its published MD5 plus record SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

root=Path('data')
root.mkdir(exist_ok=True)
archive=root/'scifact.zip'
url='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
if not archive.exists():
    urllib.request.urlretrieve(url,archive)
raw=archive.read_bytes()
if hashlib.md5(raw).hexdigest()!='5f7d1de60b170fc8027bb7898e2efca1':
    raise ValueError('Archive differs from published BEIR checksum')
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if not (root/name).resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe archive member')
    z.extractall(root)
(root/'scifact-source.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(raw).hexdigest()},indent=2))
print('Verified archive:',root/'scifact')
