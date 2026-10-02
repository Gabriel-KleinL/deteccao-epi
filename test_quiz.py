import json
import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from quiz_servidor import Quiz, GameError, QUESTIONS, QUESTIONS_PER_GAME, public_origin, load_host_key, DEFAULT_AVATAR


class HostKeyTest(unittest.TestCase):
    def test_saved_key_survives_restart_and_keeps_leading_zero(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path=Path(folder)/'quiz-host-key'
            path.write_text('012345\n')
            for _ in range(2):
                self.assertEqual(load_host_key(path),'012345')
            quiz=Quiz(Path(folder)/'quiz.sqlite3',load_host_key(path))
            self.assertEqual(quiz.create({'key':'012345'})['role'],'host')
            with self.assertRaises(GameError):quiz.create({'key':'12345'})

    def test_first_start_saves_private_key_without_changing_it_next_time(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path=Path(folder)/'nested/quiz-host-key'
            first=load_host_key(path)
            self.assertTrue(first)
            self.assertEqual(load_host_key(path),first)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_environment_key_takes_priority_without_overwriting_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'QUIZ_HOST_KEY':'configured-key'}):
            path=Path(folder)/'quiz-host-key'
            path.write_text('saved-key\n')
            self.assertEqual(load_host_key(path),'configured-key')
            self.assertEqual(path.read_text(),'saved-key\n')

    def test_empty_saved_key_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path=Path(folder)/'quiz-host-key';path.write_text(' \n')
            with self.assertRaises(ValueError):load_host_key(path)


class PublicURLTest(unittest.TestCase):
    def test_origin_for_public_links(self):
        self.assertEqual(public_origin('https://epi.in9automacao.com.br/'),
                         'https://epi.in9automacao.com.br')
        self.assertEqual(public_origin('http://localhost:8081'), 'http://localhost:8081')
        for value in ('ftp://example.org', '//example.org', 'https://',
                      'https://user:secret@example.org', 'https://example.org/quiz',
                      'https://example.org?key=secret', 'https://example.org#secret'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                public_origin(value)


class QuestionBankTest(unittest.TestCase):
    def test_thirty_distinct_questions_with_four_answers(self):
        self.assertEqual(len(QUESTIONS),30)
        self.assertEqual(len({q['question'].strip().casefold() for q in QUESTIONS}),30)
        for q in QUESTIONS:
            with self.subTest(question=q['question']):
                self.assertTrue(q['question'].strip())
                self.assertEqual(len(q['options']),4)
                self.assertTrue(all(isinstance(option,str) and option.strip() for option in q['options']))
                self.assertEqual(len(set(q['options'])),4)
                self.assertIs(type(q['correct']),int)
                self.assertIn(q['correct'],range(4))
                self.assertTrue(q['explanation'].strip())
                if 'image' in q:
                    self.assertIs(q['image'],True)


class QuizTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        self.quiz = Quiz(Path(self.tmp.name)/'quiz.sqlite3','teacher-key',lambda:self.now)
        self.host = self.quiz.create({'key':'teacher-key','seconds':25})
        self.pin = self.host['pin']
        with self.quiz.connect() as db:
            self.questions = json.loads(db.execute('SELECT questions FROM rooms WHERE pin=?', (self.pin,)).fetchone()[0])
        self.players = [self.quiz.join({'pin':self.pin,'name':f'Aluno {i}'}) for i in range(3)]

    def state(self, session=None):
        return self.quiz.state(self.pin,(session or self.host)['token'])

    def next(self, wait=True):
        s=self.state()
        self.quiz.action(self.pin,self.host['token'],{'action':'next','phase':s['phase'],'idx':s['idx']})
        if wait and self.state()['phase']=='ranking':self.now+=5
        if wait and self.state()['phase']=='reading':self.now+=5

    def answer(self, player, choice, idx=0):
        return self.quiz.answer(self.pin,player['token'],{'idx':idx,'choice':choice})

    def test_room_has_ten_unique_questions_from_the_bank(self):
        self.assertEqual(len(self.questions),10)
        self.assertEqual(len({q['question'] for q in self.questions}),10)
        self.assertTrue(all(q in QUESTIONS for q in self.questions))
        for session in [self.host,*self.players]:
            self.assertEqual(self.state(session)['total'],10)

    def test_new_room_draws_a_fresh_sample_and_keeps_its_order(self):
        selections=[QUESTIONS[::3],list(reversed(QUESTIONS[1::3]))]
        with patch('quiz_servidor.secrets.SystemRandom') as random:
            random.return_value.sample.side_effect=selections
            for selected in selections:
                host=self.quiz.create({'key':'teacher-key'})
                with self.quiz.connect() as db:
                    saved=json.loads(db.execute('SELECT questions FROM rooms WHERE pin=?',(host['pin'],)).fetchone()[0])
                self.assertEqual(saved,selected)
                self.assertEqual(self.quiz.state(host['pin'],host['token'])['total'],10)
            self.assertEqual(random.return_value.sample.call_count,2)
            random.return_value.sample.assert_called_with(QUESTIONS,QUESTIONS_PER_GAME)

    def test_selection_survives_bank_changes_reconnect_and_restart(self):
        self.next()
        self.answer(self.players[0],self.questions[0]['correct'])
        before=self.state()
        # Recarregar o banco não pode alterar perguntas, ordem ou respostas da sala.
        with patch('quiz_servidor.QUESTIONS',list(reversed(QUESTIONS))), \
                patch('quiz_servidor.secrets.SystemRandom',side_effect=AssertionError('Novo sorteio indevido')):
            restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
            self.assertEqual(restored.state(self.pin,self.host['token']),before)
            for p in self.players:
                self.assertEqual(restored.state(self.pin,p['token']),self.state(p))
            with restored.connect() as db:
                saved=json.loads(db.execute('SELECT questions FROM rooms WHERE pin=?',(self.pin,)).fetchone()[0])
            self.assertEqual(saved,self.questions)

    def test_legacy_room_keeps_original_ten_questions(self):
        legacy=QUESTIONS[:10]
        with self.quiz.connect() as db:
            db.execute('UPDATE rooms SET questions=? WHERE pin=?',(json.dumps(legacy),self.pin))
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','teacher-key',lambda:self.now)
        restored.create({'key':'teacher-key'})
        self.next()
        self.assertEqual(self.state()['total'],10)
        self.assertEqual(self.state()['question']['question'],legacy[0]['question'])
        with restored.connect() as db:
            saved=json.loads(db.execute('SELECT questions FROM rooms WHERE pin=?',(self.pin,)).fetchone()[0])
        self.assertEqual(saved,legacy)

    def test_five_seconds_reading_then_twenty_five_to_answer(self):
        self.next(wait=False)
        host=self.state()
        self.assertEqual(host['phase'],'reading')
        self.assertEqual(host['remaining'],5)
        self.assertEqual(host['question']['question'],self.questions[0]['question'])
        self.assertNotIn('options',host['question'])
        for instant in (1000,1004.999):
            self.now=instant
            self.assertEqual(self.state(self.players[0])['question'],{})
            with self.assertRaises(GameError):self.answer(self.players[0],0)
        self.now=1002
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
        self.assertEqual(restored.state(self.pin,self.players[0]['token'])['remaining'],3)
        self.now=1005
        host=self.state()
        student=self.state(self.players[0])
        self.assertEqual(host['phase'],'question')
        self.assertEqual(host['remaining'],25)
        self.assertEqual(host['question']['options'],self.questions[0]['options'])
        self.assertEqual(student['question'],{'options':self.questions[0]['options']})
        self.answer(self.players[0],self.questions[0]['correct'])
        self.now=1029.999
        self.answer(self.players[1],self.questions[0]['correct'])
        self.now=1030
        with self.assertRaises(GameError):self.answer(self.players[2],0)
        student=self.state(self.players[0])
        self.assertEqual(student['phase'],'reveal')
        self.assertEqual(student['my_answer']['points'],1000)
        self.assertEqual(self.state(self.players[1])['my_answer']['points'],500)
        self.assertEqual(student['question']['options'],self.questions[0]['options'])
        self.assertNotIn('question',student['question'])
        self.assertNotIn('explanation',student['question'])
        self.next(wait=False)
        self.assertEqual(self.state()['phase'],'ranking')
        self.now+=5
        self.assertEqual(self.state()['phase'],'reading')
        self.assertEqual(self.state()['remaining'],5)
        self.assertEqual(self.state()['question'].get('image',False),self.questions[1].get('image',False))
        self.assertNotIn('options',self.state()['question'])

    def test_new_rooms_always_allow_twenty_five_seconds_to_answer(self):
        for previous_setting in (15,20,25,30,45,60):
            host=self.quiz.create({'key':'teacher-key','seconds':previous_setting})
            self.assertEqual(self.quiz.state(host['pin'],host['token'])['seconds'],25)
        host=self.quiz.create({'key':'teacher-key'})
        self.assertEqual(self.quiz.state(host['pin'],host['token'])['seconds'],25)

    def test_ranking_between_rounds_then_reading_then_answering(self):
        self.next()
        for p in self.players:self.answer(p,self.questions[0]['correct'])
        self.next(wait=False)
        started=self.now
        for session in [self.host,*self.players]:
            s=self.state(session)
            self.assertEqual(s['phase'],'ranking')
            self.assertEqual(s['idx'],0)
            self.assertEqual(s['remaining'],5)
            self.assertIsNone(s['question'])
            self.assertNotIn('distribution',s)
        for offset in (0,4.999):
            self.now=started+offset
            with self.assertRaises(GameError):self.answer(self.players[0],0,1)
            with self.assertRaises(GameError):self.quiz.action(self.pin,self.host['token'],{'action':'next','phase':'ranking','idx':0})
        self.now=started+5
        self.assertEqual(self.state()['phase'],'reading')
        self.assertEqual(self.state()['idx'],1)
        self.assertEqual(self.state()['remaining'],5)
        self.assertNotIn('options',self.state()['question'])
        self.now=started+10
        self.assertEqual(self.state()['phase'],'question')
        self.assertEqual(self.state()['remaining'],25)
        self.now+=25
        self.assertEqual(self.state()['phase'],'reveal')

    def test_real_rank_changes_survive_restart_and_preserve_ties(self):
        # Rodada anterior: A=1000, B=800, C=600. Atual: A erra, B e C somam 1000.
        from quiz_servidor import digest
        with self.quiz.connect() as db:
            for i,p in enumerate(self.players):
                db.execute('UPDATE players SET joined=? WHERE token=?',(1000+i,digest(p['token'])))
                db.execute('INSERT INTO answers VALUES(?,?,?,?,?)',(digest(p['token']),0,0,1000-i*200,0))
            db.execute("UPDATE rooms SET phase='question',idx=1,started=?,seconds=25 WHERE pin=?",(self.now,self.pin))
        correct=self.questions[1]['correct']
        self.answer(self.players[0],(correct+1)%4,1)
        self.answer(self.players[1],correct,1)
        self.answer(self.players[2],correct,1)
        self.next(wait=False)
        s=self.state()
        self.assertEqual([p['name'] for p in s['leaderboard']],['Aluno 1','Aluno 2','Aluno 0'])
        self.assertEqual([p['movement'] for p in s['leaderboard']],[1,1,-2])
        self.assertEqual([p['previous_position'] for p in s['leaderboard']],[1,2,0])
        self.assertEqual([p['previous_score'] for p in s['leaderboard']],[800,600,1000])
        self.assertEqual([p['score'] for p in s['leaderboard']],[1800,1600,1000])
        self.assertEqual([p['gained'] for p in s['leaderboard']],[1000,1000,0])
        self.now+=2
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
        again=restored.state(self.pin,self.host['token'])
        self.assertEqual(again['remaining'],3)
        self.assertEqual(again['leaderboard'],s['leaderboard'])
        for p in self.players:
            personal=restored.state(self.pin,p['token'])
            self.assertEqual(sum(row['me'] for row in personal['leaderboard']),1)
            self.assertTrue(all('token' not in row for row in personal['leaderboard']))
        # Empates usam a mesma colocação, com ordem estável entre reconexões.
        with self.quiz.connect() as db:
            db.execute('UPDATE answers SET points=800 WHERE player=? AND idx=1',(digest(self.players[1]['token']),))
        tied=self.state()['leaderboard']
        self.assertEqual([p['rank'] for p in tied],[1,1,3])
        self.assertEqual([p['name'] for p in tied[:2]],['Aluno 1','Aluno 2'])

    def test_late_reconnect_keeps_scheduled_round_deadlines(self):
        self.next()
        for p in self.players:self.answer(p,self.questions[0]['correct'])
        self.next(wait=False)
        self.now+=12
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
        s=restored.state(self.pin,self.host['token'])
        self.assertEqual(s['phase'],'question')
        self.assertEqual(s['idx'],1)
        self.assertEqual(s['remaining'],23)

    def test_last_ranking_leads_to_final_results_without_extra_question(self):
        with self.quiz.connect() as db:
            db.execute("UPDATE rooms SET phase='reveal',idx=9 WHERE pin=?",(self.pin,))
        self.next(wait=False)
        self.assertEqual(self.state()['phase'],'ranking')
        self.assertIsNone(self.state()['question'])
        self.now+=5
        self.assertEqual(self.state()['phase'],'finished')
        self.assertEqual(self.state()['idx'],9)

    def test_existing_round_preserved_and_next_uses_new_timing(self):
        # Uma rodada antiga em andamento mantém seu prazo ao reconectar.
        with self.quiz.connect() as db:
            db.execute("UPDATE rooms SET phase='question',idx=0,seconds=20,started=1000 WHERE pin=?",(self.pin,))
        self.now=1005
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
        self.assertEqual(restored.state(self.pin,self.players[0]['token'])['remaining'],15)
        self.now=1020
        self.assertEqual(self.state()['phase'],'reveal')
        self.next(wait=False)
        self.assertEqual(self.state()['phase'],'ranking')
        self.now+=5
        self.assertEqual(self.state()['phase'],'reading')
        self.assertEqual(self.state()['remaining'],5)
        self.assertEqual(self.state()['seconds'],25)

    def test_complete_game_and_reconnect(self):
        for i,q in enumerate(self.questions):
            self.next()
            self.assertEqual(self.state(self.players[0])['question'],{'options':q['options']})
            self.assertEqual(self.state()['question']['options'],q['options'])
            self.now+=2
            for p in self.players:
                self.answer(p,q['correct'],i)
            self.assertEqual(self.state()['phase'],'reveal')
        self.next()
        restored=Quiz(Path(self.tmp.name)/'quiz.sqlite3','new-key',lambda:self.now)
        s=restored.state(self.pin,self.players[0]['token'])
        self.assertEqual(s['phase'],'finished')
        self.assertEqual(s['total'],10)
        self.assertEqual(s['idx'],9)
        self.assertEqual(s['leaderboard'][0]['score'],9600)
        self.assertTrue(all(p['rank']==1 for p in s['leaderboard']))
        self.assertEqual(s['leaderboard'][0]['correct'],10)

    def test_no_answer_or_score_leak(self):
        self.next();self.answer(self.players[0],self.questions[0]['correct'])
        for session in [self.host,*self.players]:
            s=self.state(session)
            self.assertNotIn('correct',s['question'])
            self.assertNotIn('explanation',s['question'])
            self.assertNotIn('distribution',s)
            self.assertTrue(all(p['score']==0 for p in s['leaderboard']))
        self.assertNotIn('points',self.state(self.players[0])['my_answer'])

    def test_deadline_and_duplicate(self):
        self.next();self.answer(self.players[0],0)
        with self.assertRaises(GameError):self.answer(self.players[0],1)
        self.now+=25
        with self.assertRaises(GameError):self.answer(self.players[1],0)
        self.assertEqual(self.state()['phase'],'reveal')
        self.next()
        with self.assertRaises(GameError):self.answer(self.players[1],0,0)

    def test_host_authority_and_nicknames(self):
        with self.assertRaises(GameError):self.quiz.create({'key':'wrong'})
        with self.assertRaises(GameError):self.quiz.join({'pin':self.pin,'name':'ALUNO 0'})
        with self.assertRaises(GameError):self.quiz.action(self.pin,self.players[0]['token'],{'action':'next','idx':-1,'phase':'lobby'})
        with self.assertRaises(GameError):self.quiz.state(self.pin,'wrong-token')
        self.next()
        with self.assertRaises(GameError):self.quiz.join({'pin':self.pin,'name':'Tarde'})

    def test_concurrent_duplicate_and_advance(self):
        self.next()
        def answer(_):
            try:self.answer(self.players[0],0);return True
            except GameError:return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            self.assertEqual(sum(pool.map(answer,range(10))),1)
        self.now+=26
        self.state()
        def advance(_):
            try:self.quiz.action(self.pin,self.host['token'],{'action':'next','idx':0,'phase':'reveal'});return True
            except GameError:return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            self.assertEqual(sum(pool.map(advance,range(10))),1)
        self.assertEqual(self.state()['phase'],'ranking')
        self.assertEqual(self.state()['idx'],0)
        self.now+=5
        self.assertEqual(self.state()['idx'],1)

    def test_avatar_validation_persistence_and_legacy_participants(self):
        avatar={'character':29,'helmet':False,'glasses':True,'mask':True,'vest':False}
        player=self.quiz.join({'pin':self.pin,'name':'Avatar personalizado','avatar':avatar})
        own=self.state(player)['leaderboard']
        self.assertEqual(len(own),1)
        self.assertEqual(own[0]['avatar'],avatar)
        restored=Quiz(self.quiz.db_path,'teacher-key',lambda:self.now)
        self.assertEqual(restored.state(self.pin,player['token'])['leaderboard'][0]['avatar'],avatar)
        for invalid in [[],{'character':0},{**avatar,'character':True},{**avatar,'character':30},
                        {**avatar,'helmet':'false'},{**avatar,'vest':1},{**avatar,'image':'<svg onload=alert(1)>'}]:
            with self.subTest(avatar=invalid),self.assertRaises(GameError):
                self.quiz.join({'pin':self.pin,'name':'Inválido','avatar':invalid})
        self.assertEqual(self.state()['players'],4)
        with self.quiz.connect() as db:
            before={t:[tuple(row) for row in db.execute('SELECT * FROM '+t+' ORDER BY 1,2')] for t in ('rooms','players','answers')}
            db.execute('DROP TABLE player_avatars')
        legacy=Quiz(self.quiz.db_path,'teacher-key',lambda:self.now)
        with legacy.connect() as db:
            after={t:[tuple(row) for row in db.execute('SELECT * FROM '+t+' ORDER BY 1,2')] for t in before}
        self.assertEqual(before,after)
        self.assertTrue(all(p['avatar']==DEFAULT_AVATAR for p in legacy.state(self.pin,self.host['token'])['leaderboard']))

    def test_student_only_receives_own_ranking_and_host_never_has_me_marker(self):
        def check():
            host=self.state()
            self.assertEqual(len(host['leaderboard']),3)
            self.assertFalse(any(p['me'] for p in host['leaderboard']))
            for i,session in enumerate(self.players):
                student=self.state(session)
                self.assertEqual(student['players'],3)
                self.assertEqual([p['name'] for p in student['leaderboard']],[f'Aluno {i}'])
                self.assertTrue(student['leaderboard'][0]['me'])
                self.assertNotIn('question_stats',student)
                self.assertNotIn('distribution',student)
        check()
        self.next(False);check()
        self.now+=5;check()
        for p in self.players:self.answer(p,self.questions[0]['correct'])
        check()
        self.next(False);check()
        with self.quiz.connect() as db:db.execute("UPDATE rooms SET phase='finished',idx=9 WHERE pin=?",(self.pin,))
        check()

    def test_choice_distribution_per_question_counts_zero_and_unanswered(self):
        self.next(False)
        self.assertNotIn('question_stats',self.state())
        self.now+=5
        self.answer(self.players[0],0);self.answer(self.players[1],1)
        self.assertNotIn('distribution',self.state())
        self.now+=25
        self.assertEqual(self.state()['distribution'],[1,1,0,0])
        self.next();self.answer(self.players[0],3,idx=1)
        self.now+=25
        self.assertEqual(self.state()['distribution'],[0,0,0,1])
        self.next();self.now+=25
        self.assertEqual(self.state()['distribution'],[0,0,0,0])
        self.assertNotIn('distribution',self.state(self.players[0]))

    def test_expired_room_cleanup_also_removes_only_its_avatars(self):
        self.now+=86401
        self.quiz.create({'key':'teacher-key'})
        with self.quiz.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM player_avatars').fetchone()[0],0)

    def test_independent_rooms_and_wrong_answer(self):
        other=self.quiz.create({'key':'teacher-key','seconds':25})
        p=self.quiz.join({'pin':other['pin'],'name':'Aluno 0'})
        with self.assertRaises(GameError):self.quiz.state(self.pin,p['token'])
        wrong=(self.questions[0]['correct']+1)%4
        self.next();self.answer(self.players[0],wrong)
        self.now+=26
        s=self.state(self.players[0])
        self.assertEqual(s['my_answer']['points'],0)
        self.assertNotIn('distribution',s)
        self.assertEqual(self.state()['distribution'],[int(i==wrong) for i in range(4)])
        self.assertEqual(self.quiz.state(other['pin'],other['token'])['phase'],'lobby')


if __name__=='__main__':unittest.main()
