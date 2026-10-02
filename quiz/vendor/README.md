# QR Code local

Biblioteca `qrcode-generator`, de Kazuhiko Arase, licença MIT.

Fonte: https://github.com/kazuhikoarase/qrcode-generator/tree/master/js
Arquivo obtido em 14/09/2026: https://raw.githubusercontent.com/kazuhikoarase/qrcode-generator/master/js/dist/qrcode.js
SHA-256 da cópia incluída: `79ec86f82856005b1c887905cfccfcfbec3821ca61c7fd5a952faa5f778f791c`.

A licença completa está em `LICENSE.qrcode.txt`; os créditos originais foram mantidos.
O servidor entrega esta biblioteca antes do código de `quiz/app.js` na mesma rota
`/quiz/app.js`. Não requer CDN, novas rotas no proxy nem API externa de QR Code.
O QR é SVG com margem de quatro módulos e correção M. O conteúdo é somente o
link do aluno e, na sala, o PIN; nunca contém chave do professor ou token de sessão.
