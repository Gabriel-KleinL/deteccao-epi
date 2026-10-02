# Detecção de EPI em Canteiros de Obras

Esse projeto surgiu da necessidade de identificar automaticamente se trabalhadores em canteiros de obras estão usando os equipamentos de proteção individual corretos. A ideia é simples: apontar uma câmera para o ambiente e o sistema avisa quem está sem capacete, sem máscara ou sem colete.

O modelo foi treinado com um conjunto de dados do Roboflow contendo 2801 imagens de canteiros de obras reais, divididas em treino, validação e teste. O treinamento durou 100 épocas em uma GPU P100 e levou cerca de 2 horas e meia.

São detectadas 10 categorias: capacete, máscara, colete de segurança, cone de segurança, pessoa, maquinário, veículo, e as versões negativas de cada EPI (sem capacete, sem máscara, sem colete).


## Como instalar

Você precisa ter o Python 3.10 ou superior instalado. Depois, instale as dependências com o comando abaixo. Funciona no Windows e no Mac.

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


## Melhorando e retreinando o modelo de óculos

O modelo em `modelos/oculos.pt` separa duas situações: proteção ocular
confirmada e proteção ocular ausente/inadequada. Óculos de grau ou óculos de
sol não devem ser tratados como EPI. Não use o nome da captura automática como
rótulo verdadeiro: ele é apenas a previsão do modelo antigo.

Antes de qualquer treino, audite o conjunto de dados:

    python auditar_dataset.py --data dados/oculos.yaml

Para preservar e selecionar uma amostra diversa das capturas reais, sem mexer
nos arquivos originais:

    python preparar_rotulacao.py --limite 400

Rotule manualmente o lote criado em `dados/rotulacao_pendente`. Todas as áreas
oculares identificáveis devem ser anotadas; rostos pequenos, ocultos ou sem
informação suficiente não devem receber um rótulo por suposição. Ao montar o
novo dataset, separe treino, validação e teste por vídeo/dia/câmera, nunca por
frames vizinhos.

Depois de juntar os dados antigos com os novos dados reais, atualize os caminhos
em `dados/oculos.yaml` (ou passe outro YAML em `--data`) e gere um candidato:

    python treinar_oculos.py --device auto

O treino grava pesos versionados em `modelos/candidatos` e compara as duas
classes nos splits de validação e teste. Ele não substitui automaticamente o
modelo em produção. Um candidato só é considerado apto offline quando passa
nos dois splits; a promoção só deve acontecer após medi-lo em um lote de campo
rotulado, separado do treino.

Observação: o dataset principal de capacete, máscara e colete não está presente
neste repositório; apenas os pesos treinados estão disponíveis. Para retreinar
essas classes também, primeiro é necessário restaurar/exportar esse dataset e
adicionar imagens do ambiente real de instalação.


## Apresentação

Abra [apresentacao-escola.html](apresentacao-escola.html) no navegador. São onze
slides: conteúdo, demonstrações didáticas e um acesso ao quiz separado no final.
No slide 8, a resposta só aparece ao clicar em **Mostrar resposta**.

No Mac configurado, clique em **Abrir o sistema** no slide 9. O aplicativo
**I9 EPI Local** inicia o servidor se necessário, aguarda ficar pronto e abre
`http://localhost:8080/interface.html`. Se já estiver rodando, reutiliza a instância.
A webcam e os modelos são usados no próprio Mac, e o terminal não precisa ficar
aberto. O Safari pode pedir confirmação para abrir o aplicativo e usar a câmera.
O cabeçalho da apresentação contém somente o botão **Tela cheia**.
O slide 11 continua abrindo o quiz público.
Na interface, **Óculos de Proteção**, **Sem Óculos de Proteção** e **Sem Máscara**
começam desligados. Os controles continuam disponíveis para ativação manual.

Para configurar outro Mac com as dependências instaladas:

    /opt/homebrew/bin/python3 macos/instalar.py --registrar

Isso cria `.macos/I9 EPI Local.app` dentro do projeto e registra somente o
protocolo `i9-epi://abrir`. Se mover a pasta do projeto, execute novamente.
O iniciador aceita apenas esse endereço fixo e não aceita comandos pelo link.
Também é possível dar dois cliques em `iniciar-mac.command`.
O servidor roda em segundo plano, apenas em loopback nas portas 8080/8765;
ele não é configurado para iniciar no login. O log fica em `logs/servidor-mac.log`.
Essa abertura define `PRESERVAR_HISTORICO=1`, evitando a limpeza de logs e capturas
na inicialização. O modo habitual de execução mantém a política configurada.

Para remover a associação local antes de apagar o aplicativo:

    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -u "$PWD/.macos/I9 EPI Local.app"

## Quiz ao vivo para a turma

O quiz fica separado da apresentação e contém somente o jogo.

Para iniciar o quiz de IA e EPI, sem abrir câmera ou carregar modelos:

    python3 quiz_servidor.py

O terminal informa o link privado do professor e o endereço dos alunos na rede
Wi-Fi. O jogo tem PIN de sala, dez perguntas, cronômetro, pontuação e placar.
Veja as instruções em [quiz/README.md](quiz/README.md).
