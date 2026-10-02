"""Atualiza somente a capa e incorpora os recursos para manter o HTML offline."""
from pathlib import Path
import base64
import re

BASE = Path(__file__).resolve().parent
PAGE = BASE.parents[1] / 'apresentacao-escola.html'
html = PAGE.read_text()
style = BASE.joinpath('capa.css').read_text()
markup = BASE.joinpath('capa.html').read_text()
scene = BASE.joinpath('cena.js').read_text()
engine = '/*\n' + BASE.joinpath('LICENSE-three.txt').read_text() + '\n*/\n' + BASE.joinpath('three-r160.min.js').read_text()

if '/* CAPA-I9-3D START */' in html:
    html = re.sub(r'/\* CAPA-I9-3D START \*/.*?/\* CAPA-I9-3D END \*/', lambda _: '/* CAPA-I9-3D START */\n' + style + '\n/* CAPA-I9-3D END */', html, flags=re.S)
else:
    html = re.sub(r'/\* Original animated industrial miniature \*/.*?(?=/\* Palette sampled)', lambda _: '/* CAPA-I9-3D START */\n' + style + '\n/* CAPA-I9-3D END */\n', html, flags=re.S)

fallback = BASE.joinpath('fallback.svg')
if fallback.exists():
    src = 'data:image/svg+xml;base64,' + base64.b64encode(fallback.read_bytes()).decode()
    markup = markup.replace('id="diorama-fallback"', f'id="diorama-fallback" src="{src}"')

html, n = re.subn(r'(<section class="slide cover[^>]*>).*?(</section>)', lambda m: m[1] + '\n' + markup + m[2], html, count=1, flags=re.S)
assert n == 1, 'A capa não foi encontrada'

# A primeira versão tinha um handler próprio, removido com a ilustração antiga.
html = re.sub(r"const scenePause=\$\('scene-pause'\);.*?(?=</script>)", '', html, flags=re.S)
html = re.sub(r'<!-- CAPA-I9-SCRIPTS START -->.*?<!-- CAPA-I9-SCRIPTS END -->', '', html, flags=re.S)
bundle = '\n<!-- CAPA-I9-SCRIPTS START -->\n<script>\n' + engine.replace('</script', '<\\/script') + '\n</script>\n<script>\n' + scene + '\n</script>\n<!-- CAPA-I9-SCRIPTS END -->\n'
html = html.replace('</body>', bundle + '</body>')
PAGE.write_text(html)
print(f'Capa compilada em {PAGE} ({PAGE.stat().st_size:,} bytes)')
