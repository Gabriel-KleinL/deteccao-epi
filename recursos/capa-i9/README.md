# Capa 3D da apresentação

A cena original é construída em `cena.js`, com Three.js 0.160.1. `capa.html`
e `capa.css` definem o conteúdo e o enquadramento da capa.

Depois de editar esses arquivos, execute na raiz do projeto:

```sh
python3 recursos/capa-i9/compilar.py
```

O compilador incorpora código, biblioteca e imagem alternativa ao
`apresentacao-escola.html`. O arquivo final abre diretamente no navegador,
sem servidor, CDN ou conexão com a internet. Os slides de conteúdo não são
reescritos pelo compilador.

A maquete usa WebGL. Arraste para girar; com o canvas selecionado, use as setas.
Os controles restauram o enquadramento e pausam a animação. A renderização para
quando a capa ou a aba fica oculta, e respeita a preferência de reduzir movimento.
Sem WebGL, aparece a ilustração vetorial alternativa. Ao imprimir, uma captura
da maquete substitui o canvas.

O ciclo de 28 segundos mostra um trabalhador retirando o capacete. A câmera
gira e inclina na direção dele, emitindo um feixe vermelho com pulsos suaves
enquanto o EPI está ausente. Ele recoloca o capacete, o alerta termina e o
trabalhador volta a caminhar. O aviso é uma simulação didática, sem usar a câmera
real ou executar detecção.

Biblioteca: https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.min.js
Licença: `LICENSE-three.txt`, também incorporada ao HTML.

Verificado no Safari: capa em janela e tela cheia, movimento, pausa, retomada,
rotação, restauração do ângulo e navegação para o conteúdo. Os nove slides
posteriores à capa foram comparados por hash com a versão anterior.
