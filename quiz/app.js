'use strict';
const $=id=>document.getElementById(id);
const app=$('app');
const isHost=location.pathname==='/professor';
const storageKey=`i9-quiz-${isHost?'professor':'aluno'}`;
const escapeHTML=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let refreshVersion=0;
let session=null,state=null,lastRender='',busy=false,connected=false,deadline=0,joinBase=`${location.origin}/quiz`;
try{session=JSON.parse(sessionStorage.getItem(storageKey)||'null');}catch{}
let hostKey='';
try{hostKey=sessionStorage.getItem('i9-quiz-chave')||'';}catch{}
const hashKey=new URLSearchParams(location.hash.slice(1)).get('chave');
if(isHost&&hashKey){hostKey=hashKey;try{sessionStorage.setItem('i9-quiz-chave',hostKey);}catch{}history.replaceState(null,'',location.pathname);}
function notice(message){$('notice').textContent=message;$('notice').hidden=!message;}
function saveSession(value){session=value;try{if(value)sessionStorage.setItem(storageKey,JSON.stringify(value));else sessionStorage.removeItem(storageKey);}catch{}}
async function request(path,data){
 const response=await fetch(path,{method:data?'POST':'GET',headers:{...(data?{'Content-Type':'application/json'}:{}),...(session?{'Authorization':`Bearer ${session.token}`}:{})},body:data?JSON.stringify(data):undefined,signal:AbortSignal.timeout(7000)});
 let result;try{result=await response.json();}catch{throw new Error('O servidor não respondeu como esperado.');}
 if(!response.ok){const error=new Error(result.error||'Não foi possível continuar.');error.status=response.status;throw error;}return result;
}
const arrowIcon='<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const trophyIcon='<svg viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M12 7H5v7q0 8 10 8m13-15h7v7q0 8-10 8M12 4h16v13a8 8 0 0 1-16 0V4Zm8 21v9m-7 2h14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
function eyeArt(){return `<div class="quiz-visual" aria-hidden="true"><svg viewBox="0 0 600 220" fill="none">
<defs><linearGradient id="visor-shade" x1="100" y1="0" x2="350" y2="210" gradientUnits="userSpaceOnUse"><stop stop-color="#393638"/><stop offset="1" stop-color="#161516"/></linearGradient><pattern id="visor-grid" width="18" height="18" patternUnits="userSpaceOnUse"><path d="M18 0H0v18" stroke="#FFFFFF" stroke-opacity=".035"/></pattern></defs>
<path d="M6 182h571M25 25v171" stroke="#373234"/><path d="M12 182h26m-13-13v26m541-13h20m-10-10v20" stroke="#8C797D"/>
<g class="visual-screen"><rect x="65" y="32" width="340" height="155" rx="15" fill="url(#visor-shade)" stroke="#72666A"/><rect x="66" y="33" width="338" height="153" rx="14" fill="url(#visor-grid)"/><path d="M84 55h17m-17 0v17m302-17h-17m17 0v17M84 166h17m-17 0v-17m302 17h-17m17 0v-17" stroke="#FF373F" stroke-width="2"/>
<path d="M148 107q44-55 89 0-45 55-89 0Z" fill="#1B191A" stroke="#E7E2E3" stroke-width="3"/><g class="visual-pupil"><circle cx="193" cy="107" r="20" fill="#BF0001"/><circle cx="193" cy="107" r="11" fill="#FF343C"/><circle cx="199" cy="101" r="4" fill="white"/></g><circle class="visual-focus" cx="193" cy="107" r="48" stroke="#FF343C" stroke-opacity=".3" stroke-dasharray="3 7"/>
<path d="M278 86h68m-68 13h47m-47 13h62" stroke="#7F7478" stroke-width="3" stroke-linecap="round"/><rect x="278" y="130" width="53" height="17" rx="4" fill="#BF0001"/><text x="288" y="142" fill="#FFF" font-family="Arial" font-size="8" font-weight="700" letter-spacing="1">DE OLHO</text><circle cx="374" cy="44" r="3" fill="#FF343C"/></g>
<g class="visual-card visual-card-back"><rect x="410" y="25" width="78" height="94" rx="12" fill="#BF0001" stroke="#FF6368"/><path d="m431 73 10 10 26-28" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></g>
<g class="visual-card visual-card-front"><rect x="378" y="91" width="104" height="105" rx="13" fill="#EEEDE8" stroke="white"/><text x="400" y="116" fill="#716964" font-family="Arial" font-size="7" font-weight="700" letter-spacing="1.2">SUA VEZ</text><path d="m404 171 20-41h8l19 41m-38-14h29" stroke="#BF0001" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></g>
<path d="M505 54h14m-7-7v14M543 153h10m-5-5v10" stroke="#D4C7CC" stroke-width="2"/><circle cx="529" cy="102" r="4" stroke="#EB535F"/><text x="82" y="208" fill="#8D8588" font-family="Arial" font-size="8" letter-spacing="2">OBSERVE. PENSE. RESPONDA.</text></svg></div>`;}
function setQuizScreen(name){app.dataset.screen=name;app.dataset.role=isHost?'host':'player';}
const avatarDefaults={character:0,helmet:true,glasses:false,mask:false,vest:true};
const equipment=[['helmet','Capacete'],['glasses','Óculos'],['mask','Máscara'],['vest','Colete']];
let avatarDraft={...avatarDefaults};
try{const saved=JSON.parse(sessionStorage.getItem('i9-quiz-avatar')||'null');if(saved&&Number.isInteger(saved.character)&&saved.character>=0&&saved.character<30){avatarDraft={character:saved.character,...Object.fromEntries(equipment.map(([key])=>[key,typeof saved[key]==='boolean'?saved[key]:avatarDefaults[key]]))};}}catch{}
let characterPage=Math.floor(avatarDraft.character/5);
function avatarSVG(config=avatarDefaults){
 const a={...avatarDefaults,...config},palette=[['#D29A70','#403127','#B84244'],['#A96743','#252128','#406EAA'],['#EFC9A7','#663F2D','#598168'],['#70462F','#1F2024','#956EB0'],['#BC8058','#3D2520','#AD7B34'],['#E6B18B','#A35D38','#386F74'],['#855638','#1E252B','#B75277'],['#B9754C','#512F2B','#5F66A2'],['#EAC6A6','#80786F','#617845'],['#67412D','#262326','#BC623F']];
 const person=Number.isInteger(a.character)&&a.character>=0&&a.character<30?a.character:0;
 const [skin,hair,shirt]=palette[(person+Math.floor(person/10)*3)%10];
 const hairstyles=[
 `<path d="M35 52V38q0-20 25-20t25 20v14l-9-14q-14 4-26-3l-7 17Z"/>`,
 `<path d="M30 72V40q0-24 30-24t30 24v32L78 79V41H42v38Z"/><circle cx="36" cy="29" r="13"/><circle cx="83" cy="29" r="13"/>`,
 `<path d="M34 53V33q12-19 25-14 23-9 28 14l-9 16-3-19q-8 12-31 10l-3 13Z"/>`,
 `<path d="M33 85V40q0-22 27-22t27 22v47L76 80V40q-17 2-29-6l-4 40Z"/>`,
 `<circle cx="38" cy="36" r="14"/><circle cx="47" cy="24" r="14"/><circle cx="65" cy="22" r="16"/><circle cx="82" cy="33" r="14"/><path d="M31 38h58v25H78V40H42v23H31Z"/>`,
 `<path d="M32 43q-1-20 18-24l5-10 9 10 10-8 2 13q16 4 13 23l-13-11-33 4-7 18Z"/>`,
 `<circle cx="36" cy="19" r="12"/><circle cx="84" cy="19" r="12"/><path d="M32 57V34q28-32 56 0v23L78 41q-15-8-36 0l-7 16Z"/>`,
 `<path d="M32 79V38q0-22 28-22t28 22v41l-12 7V36H44v49Z"/><path d="M32 60q-12 25 3 35M86 60q12 25-2 35" stroke="${hair}" stroke-width="9" stroke-linecap="round"/>`,
 `<path d="M32 51V34q13-16 32-14t24 20v15l-10-9-2-12q-6 8-18 6-6-11-16-3l-4 15Z"/>`,
 `<path d="M33 55V33q-1-20 19-20 14-6 24 9 12 6 12 22v13l-11-15q-22 4-34-7l-6 20Z"/><path d="M84 26q20 4 13 29l-7 15-2-25Z"/>`
 ];
 return `<svg class="epi-avatar" viewBox="0 0 120 136" fill="none" aria-hidden="true"><rect x="3" y="3" width="114" height="130" rx="30" fill="${shirt}" fill-opacity=".13"/><g fill="${hair}">${hairstyles[person%10]}</g><path d="M14 126V111q0-25 35-28h22q35 3 35 28v15Z" fill="${shirt}"/><path d="M50 76h20v14q-10 12-20 0Z" fill="${skin}"/><ellipse cx="36" cy="54" rx="5" ry="8" fill="${skin}"/><ellipse cx="84" cy="54" rx="5" ry="8" fill="${skin}"/><path d="M38 39q22-15 44 0v21q0 24-22 24T38 60Z" fill="${skin}"/><path d="M45 48h7m16 0h7" stroke="${hair}" stroke-width="3" stroke-linecap="round"/><g fill="#272527"><circle cx="49" cy="55" r="2.5"/><circle cx="71" cy="55" r="2.5"/></g><path d="M53 70q7 5 14 0" stroke="#703E31" stroke-width="2.5" stroke-linecap="round"/>
 ${[8,12,19,24,29].includes(person)?`<path d="M39 65q5 18 21 20 16-2 21-20l-7 5-8 7H54l-9-8Z" fill="${hair}"/><path d="M52 68q8-4 16 0" stroke="${hair}" stroke-width="3" stroke-linecap="round"/>`:person%5===2?'<g fill="#AF7053" opacity=".65"><circle cx="45" cy="63" r="1"/><circle cx="49" cy="64" r="1"/><circle cx="72" cy="64" r="1"/><circle cx="76" cy="63" r="1"/></g>':''}
 ${a.vest?'<path d="m46 85-17 7-5 35h72l-5-35-17-7-14 18Z" fill="#F3AE2C"/><path d="m41 88-3 38m42-38 3 38" stroke="#FFF5CC" stroke-width="7"/><path d="M27 113h66" stroke="#FFF5CC" stroke-width="6"/><path d="M60 103v24" stroke="#9C610E" stroke-width="2"/>':''}
 ${a.glasses?'<path d="M36 51h48l-3 13H65l-5-7-5 7H39Z" fill="#CCEFFA" fill-opacity=".65" stroke="#354752" stroke-width="3"/><path d="m40 51 6 10m20-10 6 10" stroke="white" stroke-width="2" stroke-opacity=".8"/>':''}
 ${a.mask?'<path d="m39 62-5-4m47 4 5-4M42 64l18-3 18 3-3 14-15 5-15-5Z" stroke="#DEFAFB" stroke-width="2" fill="#8EC4CC"/><path d="M47 68h26m-24 5h22" stroke="#528C98" stroke-width="1.5"/>':''}
 ${a.helmet?'<path d="M33 38q0-25 27-25t27 25v4H33Z" fill="#F3BC2E" stroke="#C78A16" stroke-width="2"/><path d="M55 13h10v23H55Z" fill="#FFDD6B"/><path d="M42 26v10m36-10v10" stroke="#D29418" stroke-width="3"/><rect x="28" y="36" width="64" height="8" rx="4" fill="#FFDA69" stroke="#C78A16" stroke-width="2"/>':''}</svg>`;
}
function characterChoices(){
 return Array.from({length:5},(_,n)=>characterPage*5+n).map(i=>`<button type="button" class="character-choice ${avatarDraft.character===i?'chosen':''}" data-character="${i}" aria-label="Personagem ${i+1} de 30" aria-pressed="${avatarDraft.character===i}">${avatarSVG({...avatarDraft,character:i})}<span>${String(i+1).padStart(2,'0')}</span></button>`).join('');
}
function avatarPicker(){
 return `<fieldset class="avatar-builder"><legend>MONTE SEU AVATAR</legend><div class="avatar-studio"><div><div id="avatar-preview" class="avatar-preview">${avatarSVG(avatarDraft)}</div><span id="avatar-name" class="avatar-name">Personagem ${String(avatarDraft.character+1).padStart(2,'0')}</span></div><div class="avatar-controls"><span class="builder-label">Escolha entre 30 personagens</span><div class="character-options" id="character-options" role="group" aria-label="Personagens">${characterChoices()}</div><div class="character-nav"><button type="button" id="characters-prev" aria-label="Ver personagens anteriores">←</button><span id="character-page" aria-live="polite">${characterPage*5+1}–${characterPage*5+5} de 30</span><button type="button" id="characters-next" aria-label="Ver próximos personagens">→</button></div><span class="builder-label">Vista os equipamentos</span><div class="equipment-options">${equipment.map(([key,label])=>`<button type="button" data-equipment="${key}" class="equipment-toggle ${avatarDraft[key]?'chosen':''}" aria-pressed="${avatarDraft[key]}"><span aria-hidden="true">${avatarDraft[key]?'✓':'+'}</span>${label}</button>`).join('')}</div></div></div><p>Seu personagem acompanha você até o placar.</p></fieldset>`;
}
function bindAvatarPicker(){
 const save=()=>{try{sessionStorage.setItem('i9-quiz-avatar',JSON.stringify(avatarDraft));}catch{}$('avatar-preview').innerHTML=avatarSVG(avatarDraft);$('avatar-name').textContent='Personagem '+String(avatarDraft.character+1).padStart(2,'0');};
 const bindCharacters=()=>app.querySelectorAll('[data-character]').forEach(b=>b.onclick=()=>{avatarDraft.character=Number(b.dataset.character);save();drawCharacters();});
 const drawCharacters=()=>{$('character-options').innerHTML=characterChoices();$('character-page').textContent=`${characterPage*5+1}–${characterPage*5+5} de 30`;bindCharacters();};
 $('characters-prev').onclick=()=>{characterPage=(characterPage+5)%6;drawCharacters();};
 $('characters-next').onclick=()=>{characterPage=(characterPage+1)%6;drawCharacters();};
 bindCharacters();
 app.querySelectorAll('[data-equipment]').forEach(b=>b.onclick=()=>{const key=b.dataset.equipment;avatarDraft[key]=!avatarDraft[key];save();drawCharacters();const chosen=avatarDraft[key];b.classList.toggle('chosen',chosen);b.setAttribute('aria-pressed',String(chosen));b.innerHTML=`<span aria-hidden="true">${chosen?'✓':'+'}</span>${equipment.find(([id])=>id===key)[1]}`;});
}
const qrCache=new Map();
function studentURL(pin=''){return `${joinBase}${/^\d{6}$/.test(String(pin))?'?pin='+pin:''}`;}
function qrSVG(url){
 if(!qrCache.has(url)){const code=qrcode(0,'M');code.addData(url,'Byte');code.make();qrCache.set(url,code.createSvgTag({cellSize:4,margin:16,scalable:true,alt:'QR Code para entrar no quiz como aluno'}));}
 return qrCache.get(url);
}
function inviteCard(pin='',room=false){
 const url=studentURL(pin),local=/^http:\/\/(localhost|127\.0\.0\.1)(:|\/)/.test(url);
 return `<div class="join-invite ${room?'room-invite':''}"><div class="qr-paper">${local?'<span class="qr-loading">Preparando o link da rede…</span>':qrSVG(url)}<span>APONTE A CÂMERA DO CELULAR</span></div><div class="join-invite-copy"><p class="eyebrow">ENTRADA DOS ALUNOS</p><h3>${room?'Escaneou. Entrou.':'O desafio começa aqui.'}</h3><p>${pin?'O QR Code já preenche o PIN da sala.':'Entre pelo QR Code ou abra o link abaixo.'}</p><a class="join-url" data-join-link href="${escapeHTML(url)}" target="_blank" rel="noopener">${escapeHTML(url)}</a><button class="secondary copy-join" type="button" data-copy-url="${escapeHTML(url)}">Copiar link ${arrowIcon}</button>${/^http:\/\//.test(url)?'<small>Use a mesma rede Wi-Fi do computador.</small>':''}</div></div>`;
}
function bindInvites(){app.querySelectorAll('[data-copy-url]').forEach(button=>button.onclick=async()=>{try{await navigator.clipboard.writeText(button.dataset.copyUrl);button.textContent='Link copiado!';}catch{notice('Selecione e copie o endereço mostrado ao lado do QR Code.');}});}
function updateInvite(){if(!state&&$('entry-invite')){$('entry-invite').innerHTML=inviteCard(new URLSearchParams(location.search).get('pin')||'');bindInvites();}}
function responsePie(distribution,players,correct,options=[]){
 const counts=distribution||[0,0,0,0],total=counts.reduce((sum,n)=>sum+n,0),colors=['#B91C36','#2456C4','#F3BC2E','#176949'];
 let start=0;const stops=counts.map((n,i)=>{const from=start;start+=total?n/total*360:0;return `${colors[i]} ${from}deg ${start}deg`;});
 const summary=counts.map((n,i)=>`${'ABCD'[i]}: ${n} ${n===1?'resposta':'respostas'}`).join('; ');
 return `<div class="response-distribution"><div class="response-pie" role="img" aria-label="Distribuição das respostas. ${summary}. ${Math.max(0,players-total)} sem resposta." style="background:${total?'conic-gradient('+stops.join(',')+')':'#514A4C'}"><div><strong>${total}</strong><span>RESPOSTAS</span></div></div><div class="pie-legend">${counts.map((n,i)=>`<div class="pie-legend-row"><span class="choice-dot answer-${i}">${'ABCD'[i]}</span><span class="pie-option">${escapeHTML(options[i]||'Alternativa '+'ABCD'[i])}${i===correct?'<small>✓ Resposta correta</small>':''}</span><strong>${n}<small>${total?Math.round(n/total*100):0}%</small></strong></div>`).join('')}<p class="pie-footnote">${Math.max(0,players-total)} ${players-total===1?'pessoa não respondeu':'pessoas não responderam'} · Percentuais sobre as respostas enviadas.</p></div></div>`;
}
function questionAnalysis(s){
 const q=s.question;if(!q)return '';
 return `<section class="question-analysis"><div class="analysis-heading"><div><p class="eyebrow">A ESCOLHA DA TURMA</p><h3>Como as respostas se dividiram?</h3></div></div><div id="analysis-chart">${responsePie(s.distribution,s.players,q.correct,q.options)}</div></section>`;
}
function finalResults(s,me){
 if(!isHost)return `<section class="personal-finish"><p class="eyebrow">DESAFIO CONCLUÍDO</p><div class="finish-avatar">${avatarSVG(me?.avatar)}</div><span class="personal-tag">VOCÊ · ${escapeHTML(me?.name||'')}</span><h1>${me?.rank||'—'}º <span>lugar.</span></h1><p>Essa é a sua conquista na turma!</p><div class="personal-stats"><div><strong>${(me?.score||0).toLocaleString('pt-BR')}</strong><span>PONTOS</span></div><div><strong>${me?.correct||0}<small> / ${s.total}</small></strong><span>ACERTOS</span></div></div><p class="personal-finish-note">O Top 5 está na tela do professor.<br>Obrigado por jogar com a turma!</p><button class="primary" id="again">Entrar em outra sala ${arrowIcon}</button></section>`;
 const top=s.leaderboard.slice(0,5);
 return `${roundTop(s,'DESAFIO CONCLUÍDO')}<div class="final-heading"><div><p class="eyebrow">QUEM FICOU DE OLHO</p><h1>O Top 5 <span>da turma.</span></h1></div><span class="final-people">${s.players} participantes · ${s.total} perguntas</span></div><section class="final-top-five" aria-label="Cinco primeiros colocados">${top.map((p,i)=>`<article class="final-contender place-${i}"><span class="final-place">${p.rank}º <small>LUGAR</small></span><div class="final-avatar">${avatarSVG(p.avatar)}</div><h3>${escapeHTML(p.name)}</h3><strong>${p.score.toLocaleString('pt-BR')}</strong><span class="final-score-label">PONTOS</span><p>${p.correct} de ${s.total} acertos</p></article>`).join('')}</section><div class="room-actions finish-actions"><button class="primary" id="again"><span>Criar nova sala</span>${arrowIcon}</button></div>`;
}
function personalRanking(s){
 const me=s.leaderboard.find(p=>p.me);if(!me)return '';
 const elapsed=Math.max(0,s.ranking_seconds-s.remaining),last=s.idx+1===s.total;
 return `${roundTop(s,`RODADA ${String(s.idx+1).padStart(2,'0')} CONCLUÍDA`)}<section class="personal-ranking" style="--ranking-delay:-${elapsed}s"><p class="eyebrow">SEU LUGAR NA DISPUTA</p><div class="personal-ranking-avatar">${avatarSVG(me.avatar)}</div><span class="personal-tag">VOCÊ · ${escapeHTML(me.name)}</span><div class="personal-ranking-place"><span class="standing-old-rank">${me.previous_rank}º</span><span class="standing-new-rank">${me.rank}º</span></div><h2>${rankMovement(me)}</h2><div class="standing-personal"><span>SUA POSIÇÃO<strong>${me.previous_rank}º <span aria-hidden="true">→</span> ${me.rank}º</strong></span><p><strong data-ranking-score="${me.position}">${me.previous_score.toLocaleString('pt-BR')}</strong> pontos<b>+${me.gained.toLocaleString('pt-BR')} nesta rodada</b></p></div><div class="standings-next" role="status"><span>${last?'Resultado final':'Próxima pergunta'} em <strong id="timer">${Math.ceil(s.remaining)}s</strong></span><div class="track"><span id="time-bar"></span></div></div></section>`;
}
function entry(){
 setQuizScreen('entry');
 $('connection').textContent=isHost?'Professor · acesso restrito':'Entrada dos alunos';
 const pin=new URLSearchParams(location.search).get('pin')||'';
 app.innerHTML=`<section class="entry access-entry"><div class="intro"><div class="entry-copy"><p class="eyebrow"><i class="live-dot" aria-hidden="true"></i> UM DESAFIO PARA TODA A TURMA</p><h1>Quem está<br><span>de olho<span class="question-mark">?</span></span></h1><p class="entry-lead">Seu olhar faz a diferença.<br>Inteligência artificial e segurança entram em jogo.</p></div><div id="entry-invite">${inviteCard(pin)}</div><div class="entry-facts"><div><strong>10<span> perguntas</span></strong><p>Sorteadas entre 30.</p></div><div><strong>1.000<span> pontos</span></strong><p>O máximo por acerto.</p></div><div><strong>Ao vivo<span> com a turma</span></strong><p>Cada resposta conta.</p></div></div></div><div class="form-panel"><div class="access-label">${isHost?'ÁREA DO PROFESSOR':'ENTRADA DOS ALUNOS'}<span>${isHost?'ACESSO COM CHAVE':'VAMOS JOGAR'}</span></div><div class="form-heading"><h2>${isHost?'Reúna a turma.':'Bora jogar?'}</h2><p>${isHost?'Crie a sala e projete o QR Code para receber seus alunos.':'Digite o PIN, escolha seu apelido e monte seu avatar.'}</p></div><form id="entry-form">${isHost?`${hostKey?'':`<label for="key">Chave do professor<input id="key" type="password" autocomplete="off" required placeholder="Sua chave de acesso"></label>`}<div class="form-timing"><span>TEMPO DE CADA RODADA</span><div><p><strong>5s</strong> para ler a pergunta</p><b aria-hidden="true">+</b><p><strong>25s</strong> para responder</p></div></div><input id="seconds" type="hidden" value="25">`:`<div class="join-fields"><label for="pin">PIN da sala<input class="pin-input" id="pin" inputmode="numeric" pattern="[0-9]{6}" maxlength="6" required placeholder="000 000" value="${escapeHTML(pin)}" autocomplete="off"></label><label for="name">Seu apelido<input id="name" minlength="2" maxlength="24" required placeholder="Como a turma te chama?" autocomplete="nickname"></label></div>${avatarPicker()}`}<button class="primary" type="submit"><span>${isHost?'Criar sala':'Entrar no jogo'}</span>${arrowIcon}</button></form><p class="form-note">${isHost?'Você decide quando começar e quando avançar para a próxima pergunta.':'Sem cadastro. Seu avatar e seus EPIs acompanham você na partida.'}</p><div class="form-bottom"><span class="form-bottom-icon" aria-hidden="true">✦</span><span>Aprender também pode ser um desafio.</span></div></div></section><div class="entry-bottom"><span>COMO FUNCIONA</span><ol><li><b>01</b> Entre na sala</li><li><b>02</b> Escolha sua resposta</li><li><b>03</b> Acompanhe seu resultado</li></ol><p>Acertos valem de 500 a 1.000 pontos. A rapidez conta.</p></div>`;
 bindInvites();if(!isHost)bindAvatarPicker();
 $('entry-form').onsubmit=async e=>{e.preventDefault();const button=e.target.querySelector('button[type="submit"]');button.disabled=true;notice('');try{let value;if(isHost){hostKey=hostKey||$('key').value;value=await request('/api/create',{key:hostKey,seconds:Number($('seconds').value)});try{sessionStorage.setItem('i9-quiz-chave',hostKey);}catch{}}else value=await request('/api/join',{pin:$('pin').value,name:$('name').value,avatar:avatarDraft});saveSession(value);lastRender='';await poll();}catch(error){notice(error.message);if(error.status===403&&isHost){hostKey='';try{sessionStorage.removeItem('i9-quiz-chave');}catch{}entry();}else button.disabled=false;}};
}
function ranking(rows,limit=10){return `<div class="leaderboard"><div class="leaderboard-heading"><h3>Placar ${state.phase==='finished'?'final':'da turma'}</h3><span>PONTOS</span></div>${rows.slice(0,limit).map(p=>`<div class="rank-row ${!isHost&&p.me?'me':''}"><span class="rank-position">${p.rank}º</span><span class="player-name">${escapeHTML(p.name)}${!isHost&&p.me?' <small>você</small>':''}</span><strong>${p.score.toLocaleString('pt-BR')} <small>pts</small></strong></div>`).join('')}</div>`;}
function actionButton(){if(!isHost)return '';if(state.phase==='lobby')return `<button class="primary" id="advance" ${state.players?'':'disabled'}><span>Começar partida</span>${arrowIcon}</button>`;if(state.phase==='question')return '<button class="secondary" id="advance">Encerrar tempo e revelar</button>';if(state.phase==='reveal')return `<button class="primary" id="advance"><span>${state.idx+1===state.total?'Ver resultado final':'Próxima pergunta'}</span>${arrowIcon}</button>`;return '';}
function roundTop(s,label){return `<div class="topline"><p class="eyebrow">${label||`PERGUNTA <strong>${String(s.idx+1).padStart(2,'0')}</strong><span class="round-total"> / ${String(s.total).padStart(2,'0')}</span>`}</p><span class="badge">SALA <b>${s.pin}</b></span></div>`;}
const choiceColors=['vermelha','azul','amarela','verde'];
function answerChoices(s){
 const q=s.question,reveal=s.phase==='reveal',own=s.my_answer;
 return q.options.map((option,i)=>{
  const letter='ABCD'[i],selected=own?.choice===i;
  const label=`Alternativa ${letter}, ${choiceColors[i]}: ${option}`;
  return `<button class="answer answer-${i} ${selected?'selected':''} ${reveal?(q.correct===i?'correct':'faded'):''}" data-choice="${i}" aria-label="${escapeHTML(label)}" aria-pressed="${selected}" ${isHost||reveal||own||!connected?'disabled':''}><span class="letter">${letter}</span><span class="answer-copy">${escapeHTML(option)}${isHost&&reveal?`<span class="answer-note">${q.correct===i?'✓ Correta · ':''}${s.distribution[i]} ${s.distribution[i]===1?'resposta':'respostas'}</span>`:''}</span></button>`;
 }).join('');
}
function rankMovement(p){
 const count=Math.abs(p.movement);
 return count?`${p.movement>0?'Subiu':'Desceu'} ${count} ${count===1?'posição':'posições'}`:'Manteve a posição';
}
function animatedRanking(s){
 if(!isHost)return personalRanking(s);
 const visible=Math.min(5,s.leaderboard.length),elapsed=Math.max(0,s.ranking_seconds-s.remaining);
 const moving=s.leaderboard.some(p=>p.position!==p.previous_position);
 const me=s.leaderboard.find(p=>p.me),last=s.idx+1===s.total;
 const rows=s.leaderboard.filter(p=>p.position<visible||p.previous_position<visible);
 return `${roundTop(s,`RODADA ${String(s.idx+1).padStart(2,'0')} CONCLUÍDA`)}<section class="standings-scene" style="--ranking-delay:-${elapsed}s"><div class="standings-heading"><div><p class="eyebrow">A DISPUTA CONTINUA</p><h1>${moving?'O placar <span>mudou.</span>':'Cada ponto <span>conta.</span>'}</h1></div><span class="standings-total">TOP ${visible}<small>${s.players} ${s.players===1?'participante':'participantes'}</small></span></div><div class="standings-viewport" style="--visible:${visible}" role="list" aria-label="Classificação após a rodada">${rows.map(p=>`<div role="listitem" class="standing-row ${p.movement>0?'rising':p.movement<0?'falling':'steady'} ${p.position===0?'leader':''} ${!isHost&&p.me?'is-me':''}" style="--from:${p.previous_position};--to:${p.position}" ${p.position>=visible?'aria-hidden="true"':''} aria-label="${escapeHTML(p.name)}, ${p.rank}º lugar, ${p.score} pontos. ${rankMovement(p)}."><span class="standing-place"><span class="standing-old-rank">${p.previous_rank}º</span><span class="standing-new-rank">${p.rank}º</span></span><span class="standing-avatar" aria-hidden="true">${avatarSVG(p.avatar)}</span><span class="standing-name">${escapeHTML(p.name)}${!isHost&&p.me?'<small>VOCÊ</small>':''}</span><span class="standing-movement" title="${rankMovement(p)}"><b aria-hidden="true">${p.movement>0?'↑':p.movement<0?'↓':'–'}</b><span>${Math.abs(p.movement)||'—'}</span></span><span class="standing-points"><strong data-ranking-score="${p.position}" aria-hidden="true">${p.previous_score.toLocaleString('pt-BR')}</strong><small>${p.gained?`+${p.gained.toLocaleString('pt-BR')} nesta rodada`:'sem novos pontos'}</small></span></div>`).join('')}</div>${!isHost&&me?`<div class="standing-personal"><span>SUA POSIÇÃO<strong>${me.previous_rank}º <span aria-hidden="true">→</span> ${me.rank}º</strong></span><p>${rankMovement(me)}<b>+${me.gained.toLocaleString('pt-BR')} pontos nesta rodada</b></p></div>`:''}<div class="standings-next" role="status"><span>${last?'Resultado final':'Próxima pergunta'} em <strong id="timer">${Math.ceil(s.remaining)}s</strong></span><div class="track"><span id="time-bar"></span></div></div></section>`;
}
function render(){
 const s=state,me=isHost?null:s.leaderboard.find(p=>p.me);
 setQuizScreen(s.phase);
 if(s.phase==='lobby'){
  const shareURL=`${joinBase}?pin=${s.pin}`;
  app.innerHTML=isHost?`${roundTop(s,'A TURMA ESTÁ CHEGANDO')}<section class="lobby"><div class="lobby-invite"><h1>Todo mundo<br><span>pronto?</span></h1><p class="lead muted">Dez perguntas sorteadas.<br>Um desafio para resolver juntos.</p><div class="pin-ticket"><span>CÓDIGO DA SALA</span><div class="pin-display" aria-label="PIN ${s.pin}">${String(s.pin).split('').map(n=>`<b>${escapeHTML(n)}</b>`).join('')}</div><p>É com esse PIN que a turma entra no jogo.</p></div>${inviteCard(s.pin,true)}</div><div class="room-panel"><div class="room-heading"><div><p class="eyebrow">ESCALAÇÃO DA TURMA</p><h2><strong>${s.players}</strong> ${s.players===1?'participante':'participantes'}</h2></div><span class="room-live"><i class="live-dot" aria-hidden="true"></i> SALA ABERTA</span></div><div class="names">${s.leaderboard.map(p=>`<span class="name"><i class="player-avatar" aria-hidden="true">${avatarSVG(p.avatar)}</i>${escapeHTML(p.name)}</span>`).join('')||'<div class="empty-room"><span aria-hidden="true">+</span><h3>A primeira pessoa vem aí.</h3><p>Compartilhe o PIN e espere a turma chegar.</p></div>'}</div><div class="room-ready"><span>${s.total} perguntas</span><i aria-hidden="true">·</i><span>5s de leitura + 25s para responder</span></div>${actionButton()}<p class="small muted">Espere todo mundo entrar antes de começar.</p></div></section>`:`<section class="waiting"><p class="eyebrow">VOCÊ ESTÁ NA SALA ${s.pin}</p><div class="waiting-avatar">${avatarSVG(me?.avatar)}</div><span class="name">${escapeHTML(me?.name||'')}</span><h1>Você está<br><span>no jogo.</span></h1><p>Aguarde o professor começar.<br>A pergunta aparece na tela da turma;<br>após 5 segundos, escolha aqui a sua resposta.</p><span class="badge">${s.players} participantes <i>·</i> ${s.total} perguntas</span><p class="rules">Acertos valem de 500 a 1.000 pontos.<br>Mantenha esta página aberta para jogar.</p></section>`;
  bindInvites();
 }else if(s.phase==='ranking'){
  app.innerHTML=animatedRanking(s);
 }else if(s.phase==='finished'){
  app.innerHTML=finalResults(s,me);
  $('again').onclick=()=>{saveSession(null);state=null;lastRender='';entry();};
 }else if(s.phase==='reading'){
  const q=s.question;
  app.innerHTML=`${roundTop(s)}<div class="question-heading">${isHost?`<h2>${escapeHTML(q.question)}</h2>`:'<h1>Olhos na tela<br><span>do professor.</span></h1>'}</div><div class="reading-layout ${isHost&&q.image?'has-image':''}">${isHost&&q.image?'<img class="question-image reading-image" src="/quiz/foto.png" alt="Três trabalhadores de capacete em uma obra. A pessoa à frente usa colete laranja; as duas ao fundo não usam colete.">':''}<div class="reading-wait"><p class="eyebrow">PRIMEIRO, LEIA A PERGUNTA</p><div class="reading-clock"><span class="reading-count" id="timer">${Math.ceil(s.remaining)}</span><span>SEGUNDOS</span></div><p>Observe com atenção.<br>As alternativas aparecem em instantes.</p><div class="track"><span id="time-bar"></span></div></div></div>`;
 }else{
  const q=s.question,reveal=s.phase==='reveal',own=s.my_answer,answered=!!own;
  const feedback=reveal?(own?(own.points>0?`Acertou! +${own.points} pontos`:'Não foi desta vez. Vamos conferir?'):'Tempo encerrado. Vamos conferir?'):'';
  app.innerHTML=`${roundTop(s)}<div class="question-heading"><h2>${isHost?escapeHTML(q.question):(reveal?'Vamos conferir?':'Escolha sua resposta')}</h2>${me?`<span class="personal-score">${escapeHTML(me.name)}<b>${me.score.toLocaleString('pt-BR')} pts</b></span>`:''}</div><div class="round-bar"><span class="time" id="timer">${reveal?'✓':Math.ceil(s.remaining)+'s'}</span><div class="track"><span id="time-bar"></span></div><span class="response-count"><b>${s.answered}<span>/${s.players}</span></b> respostas</span></div><section class="question-layout ${isHost&&q.image?'with-image':''}">${isHost&&q.image?'<div class="question-photo"><img class="question-image" src="/quiz/foto.png" alt="Três trabalhadores de capacete em uma obra: a pessoa à frente usa colete laranja; as duas ao fundo não usam colete."><span>OBSERVE A IMAGEM</span></div>':''}<div class="answers">${answerChoices(s)}</div></section>${!isHost&&answered&&!reveal?'<p class="sent" role="status"><span aria-hidden="true">✓</span> Resposta enviada. Agora é com a turma!</p>':''}${reveal?`<div class="result-banner ${!isHost&&(!own||own.points===0)?'wrong':''}"><span class="result-symbol" aria-hidden="true">${isHost||own?.points>0?'✓':'↗'}</span><div><h3>${isHost?'Resposta '+escapeHTML('ABCD'[q.correct]):feedback}</h3>${isHost?`<p>${escapeHTML(q.explanation)}</p>`:'<p>Acompanhe a explicação na tela do professor.</p>'}</div></div>`:''}${isHost&&reveal?questionAnalysis(s):''}<div class="room-actions">${actionButton()}</div>`;
  app.querySelectorAll('[data-choice]').forEach(button=>button.onclick=()=>sendAnswer(Number(button.dataset.choice)));
 }
 if($('advance'))$('advance').onclick=advance;
 tick();
}
async function advance(){if(busy||!connected)return;busy=true;$('advance').disabled=true;try{await request('/api/action',{pin:session.pin,phase:state.phase,idx:state.idx,action:state.phase==='question'?'reveal':'next'});notice('');}catch(error){notice(error.message);}finally{busy=false;lastRender='';await refresh();}}
async function sendAnswer(choice){if(busy||!connected||state?.phase!=='question')return;busy=true;app.querySelectorAll('[data-choice]').forEach(b=>b.disabled=true);try{await request('/api/answer',{pin:session.pin,idx:state.idx,choice});notice('');}catch(error){notice(error.message);}finally{busy=false;lastRender='';await refresh();}}
async function refresh(){
 if(!session)return;
 const expectedSession=session,version=++refreshVersion;
 try{const s=await request(`/api/state?pin=${session.pin}`);if(session!==expectedSession||version!==refreshVersion)return;connected=true;state=s;deadline=performance.now()+s.remaining*1000;$('connection').textContent='Conectado à sala';const key=JSON.stringify({...s,remaining:0});if(key!==lastRender){lastRender=key;render();}}
 catch(error){if(session!==expectedSession||version!==refreshVersion)return;connected=false;$('connection').textContent='Reconectando…';app.querySelectorAll('[data-choice],#advance').forEach(b=>b.disabled=true);lastRender='';if(error.status===401||error.status===404){saveSession(null);entry();notice(error.message);}}
}
let pollStarted=false;
async function poll(){await refresh();if(!pollStarted){pollStarted=true;setTimeout(loop,900);}}
async function loop(){await refresh();setTimeout(loop,900);}
function tick(){
 if(!state||!['ranking','reading','question'].includes(state.phase)||!$('timer'))return;
 const reading=state.phase==='reading',rankingPhase=state.phase==='ranking',remaining=Math.max(0,(deadline-performance.now())/1000);
 $('timer').textContent=reading?(remaining>0?Math.ceil(remaining):'…'):`${Math.ceil(remaining)}s`;
 $('time-bar').style.width=`${Math.min(100,remaining/(rankingPhase?state.ranking_seconds:reading?state.reading_seconds:state.seconds)*100)}%`;
 if(rankingPhase){
  const reduceMotion=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
  const progress=reduceMotion?1:Math.max(0,Math.min(1,(state.ranking_seconds-remaining-.55)/1.9));
  const eased=1-Math.pow(1-progress,3);
  app.querySelectorAll('[data-ranking-score]').forEach(el=>{
   const row=state.leaderboard.find(p=>p.position===Number(el.dataset.rankingScore));
   if(!row)return;
   el.textContent=Math.round(row.previous_score+row.gained*eased).toLocaleString('pt-BR');
  });
 }
 if(state.phase==='question'&&remaining<=0)app.querySelectorAll('[data-choice]').forEach(b=>b.disabled=true);
}
setInterval(tick,150);
fetch('/api/info').then(r=>r.json()).then(info=>{if(['localhost','127.0.0.1'].includes(location.hostname)&&info.join_url){joinBase=info.join_url;lastRender='';updateInvite();}}).catch(()=>{});
if(session){app.innerHTML='<section class="waiting"><h2>Reconectando à sua sala…</h2></section>';poll();}else entry();

// Ferramentas opcionais do navegador usam a mesma sala e as mesmas regras da interface.
if(document.modelContext?.registerTool){
 const lifecycle=new AbortController();
 window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
 const register=tool=>{try{Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}};
 register({name:'read_quiz_room',description:'Lê a etapa e o placar da sala atual, sem revelar gabarito durante a pergunta.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:true},async execute(){if(!session)throw new Error('Entre em uma sala primeiro.');await refresh();if(!connected)throw new Error('Sem conexão com a sala.');return state;}});
 if(!isHost)register({name:'join_quiz_room',description:'Entra na sala de espera com PIN e apelido e mostra a sala ao aluno.',inputSchema:{type:'object',properties:{pin:{type:'string',pattern:'^[0-9]{6}$'},name:{type:'string',minLength:2,maxLength:24}},required:['pin','name'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},async execute(input){if(session)throw new Error('Você já está em uma sala.');if(!input||typeof input.pin!=='string'||!/^\d{6}$/.test(input.pin)||typeof input.name!=='string'||input.name.trim().length<2||input.name.length>24)throw new Error('Informe PIN e apelido válidos.');const result=await request('/api/join',input);saveSession(result);lastRender='';await poll();return {pin:session.pin,phase:state?.phase};}});
}
