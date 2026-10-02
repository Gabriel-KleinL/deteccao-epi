# i9 Quiz — IA que enxerga

Quiz multiplayer inspirado na dinâmica do Kahoot, com identidade própria e um banco
de 30 perguntas baseadas na apresentação. Cada nova sala sorteia 10 perguntas.
Não depende de conta ou assinatura do Kahoot.

## Interface De Olho

A entrada do aluno contém PIN, apelido e um avatar com 30 personagens e
EPIs selecionáveis: capacete, óculos, máscara e colete. O avatar aparece na espera,
no placar do professor e no resultado individual, e fica salvo com a participação.
O QR Code é gerado localmente; na sala, ele já inclui o PIN. Há também o link para
quem preferir digitar ou compartilhar o endereço.

Os acessos são separados: `/quiz` para alunos e `/professor` para conduzir a partida.
A entrada do aluno não oferece botão ou link para a área do professor. A chave
continua obrigatória no servidor; esconder a navegação não substitui essa proteção.

Toda a interface usa CSS e SVG locais. As animações respeitam movimento reduzido.
A biblioteca MIT do QR está em `quiz/vendor`, com a origem e a licença preservadas.

## Abrir uma sala

Na pasta do projeto:

    python3 quiz_servidor.py

O terminal mostra dois endereços:

- **Professor:** abra no computador que vai projetar a partida e informe a chave
  de acesso configurada. Compartilhe apenas o link dos alunos.
- **Alunos:** abra nos celulares conectados à mesma rede Wi-Fi do computador.

Na tela do professor, clique em **Criar sala**. Cada rodada tem **5 segundos
de leitura + 25 segundos para responder**.
Projete o QR Code da sala, compartilhe o link exibido ou informe o endereço e o PIN de seis números.
Os alunos escolhem um apelido, customizam seu avatar e aguardam o início. Clique em **Começar
partida** quando todos estiverem na sala. O professor decide quando avançar após
a explicação de cada resposta. Ao clicar em **Próxima pergunta**, a sala mostra
um placar animado por **5 segundos**, depois inicia automaticamente a leitura da
pergunta seguinte. A primeira pergunta começa diretamente pela leitura.

O quiz é independente da apresentação e contém somente o jogo. Abra-o pelo
link do professor que apareceu no terminal ou pelo botão **Vamos ao quiz** no slide 11.
O botão abre `https://epi.in9automacao.com.br/professor`, inclusive quando a
apresentação é aberta no Mac. Se solicitado, informe a chave do professor.
A chave não fica gravada na apresentação. O quiz local também pode ser aberto
diretamente em `http://localhost:8081/professor`.

## Regras

- Banco de 30 perguntas, com quatro alternativas e uma resposta correta.
- Cada nova sala sorteia 10 perguntas, sem repetição dentro da partida.
- Todos na sala seguem o mesmo sorteio e a mesma ordem. Atualizar a página ou
  reiniciar o serviço preserva essa seleção; salas anteriores mantêm suas perguntas.
- Para jogar um novo sorteio, crie uma nova sala.
- Primeiro, o professor mostra somente a pergunta e a foto (quando houver) por **5 segundos**.
- Depois, as alternativas aparecem junto à pergunta e começam **25 segundos para responder**.
- Durante a leitura, o aluno aguarda; seus botões são liberados ao começar o tempo de resposta.
- Alternativas: **A vermelha**, **B azul**, **C amarela** e **D verde** nas duas telas.
- No celular, o aluno vê a letra, a cor e o texto de cada alternativa. O enunciado
  e a foto continuam na tela do professor; durante a leitura os botões ficam ocultos.
- A pontuação considera somente os 25 segundos de resposta; os 5 segundos de leitura
  não descontam pontos. Rodadas já em andamento mantêm seu prazo ao reiniciar o serviço.
- A explicação completa aparece no professor; os alunos veem acerto/erro e pontos.
- Acerto: `arredondar(1000 - 500 × tempo_decorrido / duração)`; de 500 a 1.000.
- Erro ou ausência de resposta: zero. Não há perda de pontos acumulados.
- Uma resposta por participante por rodada; o servidor controla prazo e pontuação.
- Todos responderam ou o tempo acabou: revelar resposta e explicação.
- O professor também pode encerrar a pergunta antes do prazo.
- Entre rodadas, o placar mostra os cinco primeiros: os nomes deslizam da posição
  anterior para a nova, com setas de subida/descida e a soma dos pontos. No celular,
  o aluno vê apenas sua própria posição, mesmo fora dos cinco primeiros. O indicador
  “você” só aparece para o participante autenticado; nunca na tela do professor.
- A etapa do placar dura 5 segundos para todos; só depois começam os 5 segundos
  de leitura e os 25 de resposta. Não é possível responder nem pular essa etapa.
- A última rodada também mostra a movimentação antes do resultado final.
- No final, o professor mostra somente os cinco primeiros colocados com seus avatares.
  O celular mostra somente o avatar, posição, pontuação e acertos do próprio aluno.
- Após cada correção, o professor vê um gráfico de pizza com as escolhas em A, B, C e D.
  A legenda mostra quantidades e percentuais sobre respostas enviadas; ausências ficam
  informadas à parte. A tela final contém apenas o Top 5, sem gráfico.
  Os dados da distribuição não são enviados antes da correção nem aos alunos.
- Empates e mudanças de posição são calculados pelo servidor a partir das respostas.
  Atualizar a página ou reiniciar o serviço mantém o prazo e os resultados do placar.
- Com movimento reduzido, as posições e pontuações finais aparecem sem animação.
- Empates em pontos compartilham a colocação; a ordem visual segue a entrada na sala.
- Entrada de novos jogadores somente antes do início; até 100 por sala.
- Atualização aproximada a cada segundo. Este limite de 100 não é uma medição de carga.
- Atualizar a página na mesma aba restaura a participação; não feche a aba durante o jogo.

Duas perguntas do banco usam a foto atual: três pessoas com capacete, uma com colete
e duas sem colete. Elas podem aparecer em qualquer rodada, se forem sorteadas.
O gabarito fica no servidor e só é enviado ao revelar a rodada.
A pontuação da rodada também fica oculta até esse momento.

## Rede e dados

Usa a biblioteca padrão do Python, porta **8081**, sem iniciar câmera ou YOLO.
O firewall deve permitir conexões de entrada nessa porta. Redes Wi-Fi com
isolamento entre dispositivos podem bloquear a conexão. Para outra porta:

    python3 quiz_servidor.py --port 8091

O estado é salvo em `saidas/quiz.sqlite3`, isolado dos dados de detecção.
As salas expiram após 24 horas e são removidas do banco ao criar uma nova sala.
O banco armazena apelidos, respostas, pontuações e escolhas do avatar; não coleta e-mail.
A tabela adicional `player_avatars` preserva as tabelas anteriores; participantes de
salas antigas usam o avatar padrão. O avatar aceita apenas um personagem de 0 a 29 e
quatro valores booleanos, sem HTML, imagens externas ou URLs. As escolhas são feitas
antes da entrada na sala. O estado enviado a cada aluno contém somente seu ranking.
Os tokens são guardados como hashes no servidor e não são expostos nos placares.
Uma reinicialização preserva salas, respostas e tokens existentes. O cronômetro
continua contando durante uma interrupção. A chave para criar novas salas é fixa
entre reinicializações: `QUIZ_HOST_KEY` tem prioridade; sem essa variável, o servidor
lê `saidas/quiz-host-key`. Em uma instalação nova, ele gera e salva uma chave nesse
arquivo privado, com permissão 600. Não publique esse arquivo nem o coloque no Git.
A chave é tratada como texto, preservando zeros à esquerda, e não é impressa nos logs.

O quiz publicado usa `https://epi.in9automacao.com.br/quiz`. A configuração de
produção está em [deploy/README.md](../deploy/README.md). O processo usa
`--public-url https://epi.in9automacao.com.br` para exibir o link correto aos alunos.
Com `QUIZ_HOST_KEY` configurada, a chave permanece após reiniciar e não é impressa
no log do serviço. Apenas as rotas necessárias ficam disponíveis pelo proxy;
o banco e o arquivo de perguntas não são publicados.

## Verificação

    python3 -m unittest test_quiz -v
    node --check quiz/app.js
    node --test test_quiz_frontend.cjs

Cobre a estrutura das 30 perguntas, sorteio de 10 sem repetição, persistência da
seleção, avatar e compatibilidade com salas antigas, privacidade do ranking,
distribuição por pergunta, entrada por QR, placar intermediário, subidas/descidas, empates, partida completa,
reconexão, tempo, resposta duplicada, requisições
simultâneas, isolamento de salas e proteção do gabarito. A interface inclui suporte
opcional a WebMCP quando disponibilizado pelo navegador; não foi validado em um
navegador com esse recurso.

## Pesquisa sobre o Kahoot

Referências oficiais consultadas em 09/09/2026:

- https://kahoot.com/schools/how-it-works/
- https://support.kahoot.com/hc/en-us/articles/360039890713-Kahoot-join-How-to-join-a-Kahoot-game
- https://support.kahoot.com/hc/en-us/articles/115002303908-How-points-work

A dinâmica aproveitada é sala ao vivo, PIN, apelido, alternativas, tempo e placar.
A fórmula acima é a regra deste quiz, não uma promessa de reproduzir exatamente
as regras ou todos os modos da plataforma Kahoot.
