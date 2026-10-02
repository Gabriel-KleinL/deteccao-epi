# Publicação no prod-direto

Aplicação: `/srv/apps/deteccao-epi`.

- Apresentação: https://epi.in9automacao.com.br/apresentacao-escola.html
- Alunos: https://epi.in9automacao.com.br/quiz
- Professor: https://epi.in9automacao.com.br/professor
- Detecção: https://epi.in9automacao.com.br/interface.html

A apresentação mantém onze slides. O slide 9 usa `i9-epi://abrir` para chamar
o aplicativo **I9 EPI Local** registrado no Mac do apresentador. Ele inicia a
detecção local se necessário e abre `http://localhost:8080/interface.html`.
Execute `macos/instalar.py --registrar` nesse Mac uma vez; o iniciador fica
local e não precisa ser publicado. O slide 11 abre o quiz público.
Nenhuma chave é embutida no HTML. O cabeçalho tem apenas **Tela cheia**.

## Serviços e arquivos

O túnel `/etc/cloudflared/config.yml` encaminha apenas o hostname EPI ao Nginx
em `127.0.0.1:50063`. O arquivo `nginx.conf` vai em
`/etc/nginx/conf.d/deteccao-epi.conf` e publica uma lista restrita de arquivos.

- `deteccao-epi.service`: Python da `.venv`, HTTP 50061 e WebSocket 50062.
  O drop-in `producao.conf` em `/etc/systemd/system/deteccao-epi.service.d/`
  mantém essas portas somente em loopback.
  `CAMERA_SERVIDOR=0` usa a webcam do navegador, pois o prod-direto não tem câmera
  conectada. A interface pede permissão e envia os quadros ao servidor por WSS.
- `deteccao-epi-quiz.service`: Python padrão, HTTP 50064 em loopback.
- `/etc/deteccao-epi/quiz.env`: `QUIZ_HOST_KEY`, arquivo privado com modo 600.
- `saidas/quiz.sqlite3`: salas e respostas persistentes; não substituir no deploy.

Atualizar os arquivos de interface não exige reiniciar os serviços. Mudanças em
Python exigem reiniciar somente o serviço correspondente. Validar Nginx e o túnel
antes de recarregar sua configuração. Manter as demais rotas do túnel intactas.

## Verificação e retorno

Executar `python3 -m unittest test_quiz -v`, conferir sintaxe JavaScript, hashes
dos arquivos, URLs HTTPS, conexão WebSocket e navegação no Safari. O gabarito,
o banco, código Python e arquivos privados devem retornar 404 pelo domínio público.

Backup desta publicação: `/srv/backups/deteccao-epi/20260914T155954Z`.
Contém os arquivos anteriores da aplicação, dados existentes, configuração do
túnel e estado dos serviços. A versão anterior estava desativada. Para retornar,
parar os dois serviços EPI, restaurar os arquivos do backup, remover o serviço
novo do quiz e o Nginx dedicado, restaurar a configuração anterior do túnel,
validar e recarregar Nginx, e reiniciar o túnel. Restaurar o estado de habilitação
registrado em `services-before.json`. Preservar dados do quiz criados após o deploy.
