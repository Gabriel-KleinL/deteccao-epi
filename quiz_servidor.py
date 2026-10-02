"""Quiz da turma: Python padrão, sem câmera ou modelos. python3 quiz_servidor.py"""
import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = Path(__file__).resolve().parent
READING_SECONDS = 5
ANSWER_SECONDS = 25
RANKING_SECONDS = 5
QUESTIONS_PER_GAME = 10
QUESTIONS = json.loads((BASE / 'quiz/perguntas.json').read_text())
DEFAULT_AVATAR = {'character': 0, 'helmet': True, 'glasses': False, 'mask': False, 'vest': True}


def validate_avatar(value):
    if value is None:
        return dict(DEFAULT_AVATAR)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_AVATAR):
        raise GameError('Escolha um avatar e os EPIs disponíveis.')
    if type(value['character']) is not int or not 0 <= value['character'] < 30:
        raise GameError('Escolha um dos 30 personagens.')
    if any(type(value[item]) is not bool for item in ('helmet', 'glasses', 'mask', 'vest')):
        raise GameError('Selecione os EPIs pelos botões do avatar.')
    return dict(value)


def load_host_key(path=BASE / 'saidas/quiz-host-key'):
    """Mantém a chave entre inicializações, inclusive ao executar sem variável de ambiente."""
    configured = os.environ.get('QUIZ_HOST_KEY')
    if configured:
        return configured
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_text().strip()
    else:
        key = secrets.token_urlsafe(18)
        with os.fdopen(fd, 'w') as file:
            file.write(key + '\n')
    if not key:
        raise ValueError('O arquivo privado da chave do professor está vazio.')
    return key


def public_origin(value):
    """Aceita somente a origem HTTP(S) usada pelo proxy público."""
    url = urlparse(value)
    if (url.scheme not in ('http', 'https') or not url.hostname or
            url.username or url.password or url.path not in ('', '/') or
            url.query or url.fragment):
        raise argparse.ArgumentTypeError('Informe uma origem HTTP(S), sem caminho ou credenciais.')
    return value.rstrip('/')


class GameError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Quiz:
    def __init__(self, db_path, host_key, clock=time.time):
        self.db_path, self.host_key, self.clock = str(db_path), host_key, clock
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS rooms (
              pin TEXT PRIMARY KEY, token TEXT NOT NULL, phase TEXT NOT NULL,
              idx INTEGER NOT NULL DEFAULT -1, seconds INTEGER NOT NULL,
              started REAL, created REAL NOT NULL, questions TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS players (
              token TEXT PRIMARY KEY, pin TEXT NOT NULL, name TEXT NOT NULL,
              normalized TEXT NOT NULL, joined REAL NOT NULL,
              UNIQUE(pin, normalized));
            CREATE TABLE IF NOT EXISTS answers (
              player TEXT NOT NULL, idx INTEGER NOT NULL, choice INTEGER NOT NULL,
              points INTEGER NOT NULL, elapsed REAL NOT NULL,
              PRIMARY KEY(player,idx));
            CREATE TABLE IF NOT EXISTS player_avatars (
              player TEXT PRIMARY KEY, config TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def room(self, db, pin):
        room = db.execute('SELECT * FROM rooms WHERE pin=?', (pin,)).fetchone()
        if not room or self.clock() - room['created'] > 86400:
            raise GameError('Sala não encontrada ou expirada. Confira o PIN.', 404)
        return dict(room)

    def role(self, db, room, token):
        hashed = digest(token)
        if hmac.compare_digest(room['token'], hashed):
            return 'host', None
        player = db.execute('SELECT * FROM players WHERE token=? AND pin=?', (hashed, room['pin'])).fetchone()
        if not player:
            raise GameError('Entre na sala para continuar.', 401)
        return 'player', dict(player)

    def refresh(self, db, room):
        if room['phase'] == 'ranking' and self.clock() >= room['started']:
            if room['idx'] + 1 >= len(json.loads(room['questions'])):
                db.execute("UPDATE rooms SET phase='finished' WHERE pin=?", (room['pin'],))
                room['phase'] = 'finished'
            else:
                # Mantém os prazos da sala, inclusive após reconexão ou reinício.
                room.update(phase='reading', idx=room['idx'] + 1,
                            started=room['started'] + READING_SECONDS, seconds=ANSWER_SECONDS)
                db.execute("UPDATE rooms SET phase=?,idx=?,started=?,seconds=? WHERE pin=?",
                           (room['phase'],room['idx'],room['started'],room['seconds'],room['pin']))
        if room['phase'] == 'reading' and self.clock() >= room['started']:
            db.execute("UPDATE rooms SET phase='question' WHERE pin=?", (room['pin'],))
            room['phase'] = 'question'
        if room['phase'] == 'question' and self.clock() >= room['started'] + room['seconds']:
            db.execute("UPDATE rooms SET phase='reveal' WHERE pin=?", (room['pin'],))
            room['phase'] = 'reveal'

    def create(self, data):
        key = data.get('key', '')
        if not isinstance(key, str) or not hmac.compare_digest(key, self.host_key):
            raise GameError('Chave do professor inválida.', 403)
        seconds = ANSWER_SECONDS
        token = secrets.token_urlsafe(32)
        # Sorteia uma vez por sala; a ordem salva vale para todos e sobrevive ao reinício.
        questions = secrets.SystemRandom().sample(QUESTIONS, QUESTIONS_PER_GAME)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # Remove apenas sessões expiradas do próprio quiz.
            expired = [r[0] for r in db.execute('SELECT pin FROM rooms WHERE created<?', (self.clock()-86400,))]
            for pin in expired:
                db.execute('DELETE FROM player_avatars WHERE player IN (SELECT token FROM players WHERE pin=?)', (pin,))
                db.execute('DELETE FROM answers WHERE player IN (SELECT token FROM players WHERE pin=?)', (pin,))
                db.execute('DELETE FROM players WHERE pin=?', (pin,))
                db.execute('DELETE FROM rooms WHERE pin=?', (pin,))
            if db.execute('SELECT COUNT(*) FROM rooms').fetchone()[0] >= 100:
                raise GameError('Limite de salas atingido. Tente novamente mais tarde.', 429)
            while True:
                pin = str(secrets.randbelow(900000)+100000)
                if not db.execute('SELECT 1 FROM rooms WHERE pin=?', (pin,)).fetchone():
                    break
            db.execute('INSERT INTO rooms(pin,token,phase,seconds,created,questions) VALUES(?,?,?,?,?,?)',
                       (pin, digest(token), 'lobby', seconds, self.clock(), json.dumps(questions)))
        return {'pin': pin, 'token': token, 'role': 'host'}

    def join(self, data):
        pin = str(data.get('pin', '')).strip()
        name = data.get('name', '')
        if not isinstance(name, str):
            raise GameError('Digite seu apelido.')
        name = ' '.join(name.split())
        if not 2 <= len(name) <= 24 or any(ord(c) < 32 for c in name):
            raise GameError('Use um apelido de 2 a 24 caracteres.')
        avatar = validate_avatar(data.get('avatar'))
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            room = self.room(db, pin)
            if room['phase'] != 'lobby':
                raise GameError('Esta partida já começou. Aguarde a próxima sala.', 409)
            if db.execute('SELECT COUNT(*) FROM players WHERE pin=?', (pin,)).fetchone()[0] >= 100:
                raise GameError('Esta sala está cheia (100 participantes).', 409)
            try:
                db.execute('INSERT INTO players VALUES(?,?,?,?,?)', (digest(token), pin, name, name.casefold(), self.clock()))
                db.execute('INSERT INTO player_avatars VALUES(?,?)', (digest(token), json.dumps(avatar)))
            except sqlite3.IntegrityError:
                raise GameError('Esse apelido já está em uso. Escolha outro.', 409)
        return {'pin': pin, 'token': token, 'role': 'player'}

    def action(self, pin, token, data):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            room = self.room(db, pin)
            role, _ = self.role(db, room, token)
            if role != 'host':
                raise GameError('Só o professor controla a partida.', 403)
            self.refresh(db, room)
            if data.get('idx') != room['idx'] or data.get('phase') != room['phase']:
                raise GameError('A rodada mudou. A tela será atualizada.', 409)
            action = data.get('action')
            if action == 'next' and room['phase'] in ('lobby', 'reveal'):
                if not db.execute('SELECT 1 FROM players WHERE pin=?', (pin,)).fetchone():
                    raise GameError('Espere pelo menos um aluno entrar.')
                if room['phase'] == 'reveal':
                    db.execute("UPDATE rooms SET phase='ranking', started=? WHERE pin=?",
                               (self.clock() + RANKING_SECONDS, pin))
                else:
                    db.execute("UPDATE rooms SET phase='reading', idx=idx+1, started=?, seconds=? WHERE pin=?", (self.clock() + READING_SECONDS, ANSWER_SECONDS, pin))
            elif action == 'reveal' and room['phase'] == 'question':
                db.execute("UPDATE rooms SET phase='reveal' WHERE pin=?", (pin,))
            else:
                raise GameError('Ação indisponível nesta etapa.', 409)
        return {'ok': True}

    def answer(self, pin, token, data):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            room = self.room(db, pin)
            role, player = self.role(db, room, token)
            if role != 'player':
                raise GameError('O professor não responde nesta tela.', 403)
            self.refresh(db, room)
            if room['phase'] == 'reading':
                raise GameError('Aguarde os 5 segundos de leitura antes de responder.', 409)
            if room['phase'] != 'question' or data.get('idx') != room['idx']:
                raise GameError('O tempo desta pergunta terminou.', 409)
            q = json.loads(room['questions'])[room['idx']]
            choice = data.get('choice')
            if type(choice) is not int or not 0 <= choice < len(q['options']):
                raise GameError('Alternativa inválida.')
            elapsed = max(0, self.clock() - room['started'])
            points = round(1000 - 500 * min(1, elapsed/room['seconds'])) if choice == q['correct'] else 0
            try:
                db.execute('INSERT INTO answers VALUES(?,?,?,?,?)', (player['token'], room['idx'], choice, points, elapsed))
            except sqlite3.IntegrityError:
                raise GameError('Sua resposta já foi registrada.', 409)
            answered = db.execute('SELECT COUNT(*) FROM answers a JOIN players p ON p.token=a.player WHERE p.pin=? AND a.idx=?', (pin, room['idx'])).fetchone()[0]
            count = db.execute('SELECT COUNT(*) FROM players WHERE pin=?', (pin,)).fetchone()[0]
            if answered == count:
                db.execute("UPDATE rooms SET phase='reveal' WHERE pin=?", (pin,))
        return {'ok': True}

    def leaderboard(self, db, pin, scored_idx, player):
        rows = db.execute('''SELECT p.token,p.name,v.config avatar,COALESCE(SUM(a.points),0) score,
                 COALESCE(SUM(CASE WHEN a.points>0 THEN 1 ELSE 0 END),0) correct
                 FROM players p LEFT JOIN answers a ON a.player=p.token AND a.idx<=?
                 LEFT JOIN player_avatars v ON v.player=p.token
                 WHERE p.pin=? GROUP BY p.token ORDER BY score DESC, p.joined ASC, p.token ASC''',
                          (scored_idx,pin)).fetchall()
        result = []
        last_score, rank = None, 0
        for position, row in enumerate(rows):
            if row['score'] != last_score:
                rank = position + 1
            last_score = row['score']
            result.append({'name':row['name'], 'score':row['score'], 'correct':row['correct'],
                           'rank':rank, 'me':bool(player and row['token']==player['token']),
                           'avatar':json.loads(row['avatar']) if row['avatar'] else dict(DEFAULT_AVATAR)})
        return result

    def state(self, pin, token):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            room = self.room(db, pin)
            role, player = self.role(db, room, token)
            self.refresh(db, room)
            questions = json.loads(room['questions'])
            # Não revelar pontuação da rodada em andamento: ela denuncia a resposta correta.
            scored_idx = room['idx'] - (room['phase'] in ('reading', 'question'))
            leaderboard = self.leaderboard(db, pin, scored_idx, player)
            if room['phase'] == 'ranking':
                previous = {p['name']:(position,p) for position,p in
                            enumerate(self.leaderboard(db,pin,room['idx']-1,player))}
                for position,p in enumerate(leaderboard):
                    old_position,old = previous[p['name']]
                    p.update(position=position,previous_position=old_position,
                             previous_rank=old['rank'],previous_score=old['score'],
                             gained=p['score']-old['score'],movement=old['rank']-p['rank'])
            answers = db.execute('SELECT a.* FROM answers a JOIN players p ON p.token=a.player WHERE p.pin=? AND a.idx=?', (pin,room['idx'])).fetchall()
            state = {'pin':pin,'role':role,'phase':room['phase'],'idx':room['idx'],'total':len(questions),
                     'seconds':room['seconds'],'reading_seconds':READING_SECONDS,'ranking_seconds':RANKING_SECONDS,
                     'remaining':max(0,room['started']+(room['seconds'] if room['phase']=='question' else 0)-self.clock()) if room['phase'] in ('ranking','reading','question') else 0,
                     'players':len(leaderboard),'answered':len(answers),'leaderboard':leaderboard,'question':None,'my_answer':None}
            if room['idx'] >= 0 and room['phase'] != 'ranking':
                q = questions[room['idx']]
                state['question'] = {k:q[k] for k in ('question','image') if k in q} if role == 'host' else {}
                if room['phase'] != 'reading':
                    # As alternativas ficam legíveis nas duas telas após a leitura.
                    state['question']['options'] = q['options']
                if room['phase'] in ('reveal','finished'):
                    state['question']['correct'] = q['correct']
                    if role == 'host':
                        state['question']['explanation'] = q['explanation']
                    if role == 'host':
                        state['distribution'] = [sum(a['choice']==i for a in answers) for i in range(len(q['options']))]
                if player:
                    own = next((a for a in answers if a['player']==player['token']), None)
                    if own:
                        state['my_answer'] = {'choice':own['choice']}
                        if room['phase'] != 'question':
                            state['my_answer']['points'] = own['points']
            # O celular recebe somente a posição do participante autenticado.
            if role == 'player':
                state['leaderboard'] = [p for p in leaderboard if p['me']]
            return state


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Não registrar PIN, apelido, respostas ou tokens nos logs HTTP.

    def respond(self, code, value, mime='application/json; charset=utf-8'):
        content = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def token(self):
        value = self.headers.get('Authorization', '')
        return value[7:] if value.startswith('Bearer ') else ''

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == '/api/info':
                return self.respond(200, {'join_url': self.server.join_url})
            if url.path == '/api/state':
                pin = parse_qs(url.query).get('pin', [''])[0]
                return self.respond(200, self.server.quiz.state(pin, self.token()))
            files = {'/':'quiz/index.html','/quiz':'quiz/index.html','/professor':'quiz/index.html',
                     '/quiz/app.js':'quiz/app.js','/quiz/style.css':'quiz/style.css','/quiz/foto.png':'quiz/foto.png'}
            if url.path not in files:
                return self.respond(404, {'error':'Página não encontrada.'})
            file = BASE / files[url.path]
            content = file.read_bytes()
            if url.path == '/quiz/app.js':
                # Biblioteca local: QR funciona sem CDN e sem transmitir o link a terceiros.
                content = (BASE / 'quiz/vendor/qrcode.js').read_bytes() + b'\n;\n' + content
            self.respond(200, content, mimetypes.guess_type(file)[0] or 'application/octet-stream')
        except GameError as e:
            self.respond(e.status, {'error':str(e)})

    def do_POST(self):
        try:
            if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                raise GameError('Envie JSON.', 415)
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 4096:
                raise GameError('Solicitação inválida.', 413)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise GameError('Solicitação inválida.')
            path = urlparse(self.path).path
            if path == '/api/create':
                result = self.server.quiz.create(data)
            elif path == '/api/join':
                result = self.server.quiz.join(data)
            elif path in ('/api/action','/api/answer'):
                pin = str(data.get('pin',''))
                fn = self.server.quiz.action if path.endswith('action') else self.server.quiz.answer
                result = fn(pin, self.token(), data)
            else:
                raise GameError('Rota não encontrada.',404)
            self.respond(200,result)
        except GameError as e:
            self.respond(e.status,{'error':str(e)})
        except (ValueError, TypeError):
            self.respond(400,{'error':'Solicitação inválida.'})


def main():
    parser = argparse.ArgumentParser(description='Quiz ao vivo de IA e EPI')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--db', default=str(BASE/'saidas/quiz.sqlite3'))
    parser.add_argument('--public-url', type=public_origin, help='Origem pública do proxy, por exemplo https://epi.in9automacao.com.br')
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    key = load_host_key()
    server = ThreadingHTTPServer((args.bind,args.port),Handler)
    server.quiz = Quiz(args.db,key)
    ip='127.0.0.1'
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
            sock.connect(('8.8.8.8',80));ip=sock.getsockname()[0]
    except OSError:
        pass
    origin = args.public_url or f'http://{ip}:{args.port}'
    server.join_url = f'{origin}/quiz'
    print(f'Alunos: {server.join_url}', flush=True)
    professor = f'{args.public_url or f"http://localhost:{args.port}"}/professor'
    print(f'Professor: {professor} (use a chave de acesso configurada)', flush=True)
    print('Mantenha este terminal aberto. Ctrl+C encerra o servidor.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
