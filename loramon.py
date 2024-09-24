import random
from data import dex
import json
import re
import sim.sim as sim
from sim.player import Decision
from pkmn_logic_utils import get_U
from tools.pick_six import agent_create_team

class TrainerBase:
    def __init__(self, name) -> None:
        self.name = name
        self.team = sim.dict_to_team_set(agent_create_team())
        
        self.plan_library = {}

    def get_available_moves(self, px: sim.Player):
        self.plan_library = {}
        if not px.active_pokemon[0].fainted:
            self.plan_library['pass'] = Decision('pass', 0)
            for i, mov in enumerate(px.active_pokemon[0].moves):
                self.plan_library['use('+mov+')'] = Decision('move', i)
        for i, pok in enumerate(px.pokemon):
            if (not pok.fainted) and (not pok.species == px.active_pokemon[0].species):
                self.plan_library['switch_to('+pok.species+', '+str(i)+')'] = Decision('switch', i)
        
    def build_knowledge(self):
        pass

    def choose(self, me: sim.Player, foe: sim.Player):
        pass

class TrainerBDI(TrainerBase):
    def __init__(self, name, graph) -> None:
        super().__init__(name)

        self.beliefs = set() #facts ()
        self.desires = set() #still from pyreason
        self.intentions = set() #choosen desires

        self.kgraph = graph
    
    def build_knowledge(self):
        pass

    def choose(self, me: sim.Player, foe: sim.Player):
        self.get_available_moves(me)
        return self.plan_library[random.choice(list(self.plan_library))]

def pokemon_log_assoc(B: sim.Battle):
    for pok in B.p1.pokemon:
        pok.log = B.log
        pok.owner = B.p1.name
    for pok in B.p2.pokemon:
        pok.log = B.log
        pok.owner = B.p2.name

def do_battle(t1: TrainerBase, t2: TrainerBase):
    B = sim.Battle('single', t1.name, t1.team, t2.name, t2.team)
    MAX_TURNS = 9999
    B.logs.append(["\n" + B.p1.name + " vs " + B.p2.name])
    B.logs.append([B.p1.name + " pokemons are:"]+[pok.species for pok in B.p1.pokemon]+['\n'])
    B.logs.append([B.p2.name + " pokemons are:"]+[pok.species for pok in B.p2.pokemon]+['\n'])
    while not B.ended and B.turn<MAX_TURNS:
        # this carries log for fainted pokemons
        pokemon_log_assoc(B)
        
        # the players select their next move
        B.p1.choice = t1.choose(B.p1, B.p2)
        B.p2.choice = t2.choose(B.p2, B.p1)

        # do a turn
        sim.do_turn(B)
    
    if B.winner == 'p1':
        winner = t1
    elif B.winner == 'p2':
        winner = t2
    else:
        winner = None

    return winner, B.logs

if __name__ == "__main__":
    player1 = TrainerBDI("Juan", None)
    player2 = TrainerBDI("Pedro", None)

    winner, logs = do_battle(player1, player2)

    for log in logs:
        for caption in log:
            print(caption)
    if winner:
        print("el ganador es", winner.name)