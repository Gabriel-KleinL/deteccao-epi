const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(`${__dirname}/quiz/app.js`,'utf8').split('setInterval(tick,150);')[0];
function page(role='professor',reduced=false){
 const elements=new Map();let clock=0,scoreNodes=[];
 const el=id=>{if(!elements.has(id))elements.set(id,{style:{},dataset:{},textContent:'',innerHTML:'',querySelectorAll:()=>[]});return elements.get(id);};
 const app=el('app');let html='';
 Object.defineProperty(app,'innerHTML',{get:()=>html,set:value=>{html=value;scoreNodes=[...value.matchAll(/data-ranking-score="(\d+)"[^>]*>([^<]*)/g)].map(m=>({dataset:{rankingScore:m[1]},textContent:m[2]}));}});
 app.querySelectorAll=selector=>selector==='[data-ranking-score]'?scoreNodes:[];
 const ctx={qrcode:require('./quiz/vendor/qrcode.js'),document:{getElementById:el},location:{pathname:'/'+role,origin:'https://epi.in9automacao.com.br',hash:'',search:'?pin=123456'},sessionStorage:{getItem:()=>null,setItem(){}},URLSearchParams,history:{replaceState(){}},performance:{now:()=>clock},matchMedia:()=>({matches:reduced})};
 vm.createContext(ctx);vm.runInContext(source,ctx);
 return {ctx,el,html:()=>html,scores:()=>scoreNodes.map(p=>p.textContent),at:time=>{clock=time;vm.runInContext('tick()',ctx);},render:s=>vm.runInContext(`state=${JSON.stringify(s)};connected=true;deadline=5000;render();`,ctx)};
}
const rows=[
 {name:'Lia',rank:1,score:2200,correct:3,position:0,previous_position:2,previous_rank:3,previous_score:1200,movement:2,gained:1000},
 {name:'Caio',rank:2,score:2000,correct:2,position:1,previous_position:0,previous_rank:1,previous_score:2000,movement:-1,gained:0},
 {name:'Ana <&>',rank:3,score:1700,correct:2,position:2,previous_position:1,previous_rank:2,previous_score:1700,movement:-1,gained:0},
];
const state={pin:'123456',phase:'ranking',idx:2,total:10,players:3,answered:3,remaining:5,ranking_seconds:5,reading_seconds:5,seconds:25,leaderboard:rows,question:null,my_answer:null};
test('placar move a ordem anterior para a atual sem antecipar a próxima pergunta',()=>{
 const p=page();p.render(state);const html=p.html();
 assert(html.includes('standing-row rising leader'));
 assert(html.includes('--from:2;--to:0'));
 assert(html.includes('standing-row falling'));
 assert(html.includes('Ana &lt;&amp;&gt;'));
 assert(!html.includes('data-choice'));
 assert(!html.includes('id="advance"'));
 assert(html.includes('Próxima pergunta em'));
 assert.deepEqual(p.scores(),['1.200','2.000','1.700']);
 p.at(1500);assert(Number(p.scores()[0].replace('.',''))>1200);assert(Number(p.scores()[0].replace('.',''))<2200);
 p.at(3000);assert.deepEqual(p.scores(),['2.200','2.000','1.700']);
 assert.equal(p.el('timer').textContent,'2s');
});
test('aluno vê a própria mudança de posição e o resultado final vem após o último placar',()=>{
 const p=page('quiz');p.render({...state,idx:9,leaderboard:rows.map((row,i)=>({...row,me:i===0}))});
 assert(p.html().includes('standing-personal'));
 assert(p.html().includes('Subiu 2 posições'));
 assert(p.html().includes('Resultado final em'));
 assert(!p.html().includes('data-choice'));
});
test('movimento reduzido exibe a pontuação final imediatamente',()=>{
 const p=page('professor',true);p.render(state);
 assert.deepEqual(p.scores(),['2.200','2.000','1.700']);
});
test('participantes fora do top 5 mantêm sua posição visível no cartão pessoal',()=>{
 const p=page('quiz');const many=Array.from({length:8},(_,i)=>({name:'Pessoa '+i,rank:i+1,score:8000-i*500,correct:3,position:i,previous_position:i,previous_rank:i+1,previous_score:7500-i*500,movement:0,gained:500,me:i===7}));
 p.render({...state,players:8,leaderboard:many});
 assert(!p.html().includes('TOP 5'));
 assert(!p.html().includes('Pessoa 0'));
 assert(p.html().includes('8º <span aria-hidden="true">→</span> 8º'));
 p.at(3000);assert.deepEqual(p.scores(),['4.500']);
});
test('celular mostra as quatro alternativas com texto, letra e cor durante a resposta e a correção',()=>{
 const p=page('quiz');
 const options=['Equipamento de proteção individual','Uma câmera que identifica possíveis ausências de EPI','Revisar o alerta antes de tomar uma decisão','Capacete, colete, óculos e máscara'];
 const question={...state,phase:'question',remaining:25,question:{options},my_answer:null};
 p.render(question);
 options.forEach((option,i)=>{
  assert(p.html().includes(`<span class="answer-copy">${option}</span>`));
  assert(p.html().includes(`<span class="letter">${'ABCD'[i]}</span>`));
  assert(p.html().includes(`answer-${i}`));
 });
 assert(p.html().includes('Alternativa A, vermelha: Equipamento de proteção individual'));
 assert(!p.html().includes('letter-only'));
 assert(!p.html().includes('correct'));
 p.render({...question,phase:'reveal',question:{options,correct:2},my_answer:{choice:2,points:1000}});
 assert(p.html().includes('selected correct'));
 options.forEach(option=>assert(p.html().includes(`<span class="answer-copy">${option}</span>`)));
 assert(p.html().includes('Acertou! +1000 pontos'));
});
test('celular aguarda os cinco segundos de leitura antes de mostrar as alternativas',()=>{
 const p=page('quiz');p.render({...state,phase:'reading',question:{}});
 assert(p.html().includes('Olhos na tela'));
 assert(!p.html().includes('data-choice'));
 assert(!p.html().includes('answer-copy'));
});
test('entrada dos alunos tem avatar e QR com PIN, sem link para o professor',()=>{
 const p=page('quiz');vm.runInContext('entry()',p.ctx);
 assert(!p.html().includes('href="/professor"'));
 assert(!p.html().includes('Sou professor'));
 assert(p.html().includes('MONTE SEU AVATAR'));
 assert(p.html().includes('https://epi.in9automacao.com.br/quiz?pin=123456'));
 assert(p.html().includes('QR Code para entrar no quiz como aluno'));
 assert.equal((p.html().match(/data-character=/g)||[]).length,5);
 for(const item of ['helmet','glasses','mask','vest'])assert(p.html().includes(`data-equipment="${item}"`));
 const host=page();vm.runInContext('entry()',host.ctx);
 assert(host.html().includes('Chave do professor'));
 assert(!host.html().includes('MONTE SEU AVATAR'));
 assert(!host.html().includes('chave='));
});
test('QR usa link da rede após bootstrap e nunca localhost ou chave do professor',()=>{
 const p=page();
 vm.runInContext("joinBase='http://127.0.0.1:8081/quiz';entry()",p.ctx);
 assert(p.html().includes('Preparando o link da rede'));
 assert(!p.html().includes('QR Code para entrar'));
 vm.runInContext("joinBase='http://192.168.1.10:8081/quiz';updateInvite()",p.ctx);
 assert(p.el('entry-invite').innerHTML.includes('http://192.168.1.10:8081/quiz?pin=123456'));
 assert(!p.el('entry-invite').innerHTML.includes('professor'));
});
test('professor nunca mostra você, mesmo se um marcador pessoal chegar ao cliente',()=>{
 const p=page();p.render({...state,leaderboard:rows.map(r=>({...r,me:true}))});
 assert(!p.html().includes('VOCÊ'));
 assert(!p.html().includes('is-me'));
});
test('professor vê Top 5 no fim, aluno vê apenas seu próprio resultado',()=>{
 const many=Array.from({length:8},(_,i)=>({...rows[0],name:'Pessoa '+i,rank:i+1,me:i===7}));
 const final={...state,phase:'finished',leaderboard:many};
 const host=page();host.render(final);
 assert.equal((host.html().match(/class="final-contender /g)||[]).length,5);
 assert(!host.html().includes('Pessoa 5'));
 const student=page('quiz');student.render(final);
 assert(student.html().includes('VOCÊ · Pessoa 7'));
 assert(student.html().includes('8º'));
 assert(!student.html().includes('Pessoa 0'));
 assert(!student.html().includes('final-contender'));
});
test('gráfico distribui A B C D, trata nenhuma resposta e não aparece para alunos',()=>{
 const p=page();
 let chart=vm.runInContext("responsePie([2,1,1,0],5,1,['Primeira','Segunda','Terceira','Quarta'])",p.ctx);
 assert(chart.includes('#B91C36 0deg 180deg'));
 assert(chart.includes('#2456C4 180deg 270deg'));
 assert(chart.includes('50%'));
 assert(chart.includes('1 pessoa não respondeu'));
 assert(chart.includes('Segunda<small>✓ Resposta correta'));
 chart=vm.runInContext('responsePie([0,0,0,0],3,0)',p.ctx);
 assert(!chart.includes('NaN'));
 assert(chart.includes('3 pessoas não responderam'));
 const final={...state,phase:'finished'};
 p.render(final);
 assert(!p.html().includes('A ESCOLHA DA TURMA'));
 assert(!p.html().includes('response-pie'));
 const student=page('quiz');student.render({...state,phase:'reveal',question:{options:['A','B','C','D'],correct:0},distribution:[2,1,0,0]});
 assert(!student.html().includes('response-pie'));
});
test('galeria percorre os 30 personagens sem perder os EPIs selecionados',()=>{
 const p=page('quiz');vm.runInContext('entry()',p.ctx);
 const characters=new Set();
 for(let i=0;i<6;i++){
  const choices=vm.runInContext('characterChoices()',p.ctx);
  for(const match of choices.matchAll(/data-character="(\d+)"/g))characters.add(Number(match[1]));
  p.el('characters-next').onclick();
 }
 assert.equal(characters.size,30);
 assert.equal(Math.min(...characters),0);assert.equal(Math.max(...characters),29);
 assert.equal(vm.runInContext('characterPage',p.ctx),0);
 const variants=vm.runInContext('Array.from({length:30},(_,character)=>avatarSVG({character,helmet:false,glasses:false,mask:false,vest:false}))',p.ctx);
 assert.equal(new Set(variants).size,30);
 assert.equal(vm.runInContext('avatarDraft.helmet',p.ctx),true);
});
