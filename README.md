# Detecção de EPI em Canteiros de Obras

Esse projeto surgiu da necessidade de identificar automaticamente se trabalhadores em canteiros de obras estão usando os equipamentos de proteção individual corretos. A ideia é simples: apontar uma câmera para o ambiente e o sistema avisa quem está sem capacete, sem máscara ou sem colete.

O modelo foi treinado com um conjunto de dados do Roboflow contendo 2801 imagens de canteiros de obras reais, divididas em treino, validação e teste. O treinamento durou 100 épocas em uma GPU P100 e levou cerca de 2 horas e meia.

São detectadas 10 categorias: capacete, máscara, colete de segurança, cone de segurança, pessoa, maquinário, veículo, e as versões negativas de cada EPI (sem capacete, sem máscara, sem colete).


## Como instalar

Você precisa ter o Python 3.8 ou superior instalado. Depois, instale as dependências com o comando abaixo. Funciona no Windows e no Mac.

    pip install -r requirements.txt


## Como usar

Para rodar o sistema completo com um único comando:

    python servidor.py

Isso abre automaticamente as duas telas no navegador: a interface da câmera com os controles de detecção e a tela do supervisor com os alertas em tempo real. Se a câmera não for a padrão, passe o índice como argumento:

    python servidor.py 1

Para rodar detecção em imagens ou vídeos salvos sem o servidor:

    python detectar.py imagem.jpg
    python detectar.py video.mp4

No Windows, se a câmera não abrir, tente o índice 1 no lugar do 0.


## Estrutura de pastas

- `modelos` — modelo treinado e modelo base do YOLOv8
- `arquivos` — imagens e vídeos de teste
- `resultados` — gráficos e métricas do treinamento
- `saidas` — resultados gerados pelo script de detecção
- `dados` — arquivo de configuração do dataset
