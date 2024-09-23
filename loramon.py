import random
from data import dex
import json
import re
import sim.sim as sim
from sim.player import Decision
from pkmn_logic_utils import get_U

with open('data/domains/all.json') as f:
    domain_all = json.load(f)
def agent_create_team(num_pokemon=6, domain='all'):
    team = []
    species = []
    items = []
    pokedex = list(domain_all)
    items = list(dex.item_dex.keys())
    natures = list(dex.nature_dex.keys())

    while len(team) < num_pokemon:
        pokemon = {}
        pokemon['species'] = random.choice(pokedex)

        moves = list(dex.simple_learnsets[pokemon['species']])
        pokemon['moves'] = random.sample(moves, min(4, len(moves)))
        
        pokemon['item'] = random.choice(items)

        pokemon['nature'] = random.choice(natures)

        abilities = [re.sub(r'\W+', '', ability.lower()) for ability in list(filter(None.__ne__, list(dex.pokedex[pokemon['species']].abilities)))]
        
        pokemon['ability'] = random.choice(abilities)

        divs = [random.randint(0,127) for i in range(5)]
        divs.append(0)
        divs.append(127)
        divs.sort()
        evs = [4*(divs[i+1]-divs[i]) if 4*(divs[i+1]-divs[i])< 252 else 252 for i in range(len(divs)-1)]
        pokemon['evs'] = evs
        pokemon['ivs'] = [31, 31, 31, 31, 31, 31]

        team.append(pokemon)

    return sim.dict_to_team_set(team)

class TrainerL:
    def __init__(self, name) -> None:
        self.name = name
        self.team = agent_create_team()
        self.current_pkmn = 0
        self.plan_library = {}

        self.beliefs = set() #facts
        self.desires = set() #still from pyreason
        self.intentions = set() #choosen desires

        self.graph = None
        self.value = 0
        
    def asynthotic_value(self):
        x = self.value
        return x/(x+1)
    
    def get_available_moves(self, px: sim.Player):
        self.plan_library = {}
        if not px.active_pokemon[0].fainted:
            self.plan_library['pass'] = Decision('pass', 0)
            for i, mov in enumerate(px.active_pokemon[0].moves):
                self.plan_library['use('+mov+')'] = Decision('move', i)
        for i, pok in enumerate(px.pokemon):
            if (not pok.fainted) and (not pok.species == px.active_pokemon[0].species):
                self.plan_library['switch_to('+pok.species+', '+str(i)+')'] = Decision('switch', i)

    def planificate(self):
        pass

    def choose(self):
        self.planificate()

        U = get_U()
        if U<self.asynthotic_value() and len(self.intentions)>0: #do rational move
            pass
        else: #select random move and learn
            pass
        return self.plan_library[random.choice(list(self.plan_library.keys()))]

def pokemon_log_assoc(B: sim.Battle):
    for pok in B.p1.pokemon:
        pok.log = B.log
        pok.owner = B.p1.name
    for pok in B.p2.pokemon:
        pok.log = B.log
        pok.owner = B.p2.name

def do_battle(t1: TrainerL, t2: TrainerL):
    B = sim.Battle('single', t1.name, t1.team, t2.name, t2.team)
    MAX_TURNS = 9999
    B.logs.append(["\n" + B.p1.name + " vs " + B.p2.name])
    B.logs.append([B.p1.name + " pokemons are:"]+[pok.species for pok in B.p1.pokemon]+['\n'])
    B.logs.append([B.p2.name + " pokemons are:"]+[pok.species for pok in B.p2.pokemon]+['\n'])
    while not B.ended and B.turn<MAX_TURNS:
        # this carries log for fainted pokemons
        pokemon_log_assoc(B)

        # update players available actions
        t1.get_available_moves(B.p1)
        t2.get_available_moves(B.p2)
        
        # the players select their next move
        B.p1.choice = t1.choose()
        B.p2.choice = t2.choose()

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
    player1 = TrainerL("Juan")
    player2 = TrainerL("Pedro")

    winner, logs = do_battle(player1, player2)

    for log in logs:
        for caption in log:
            print(caption)
    if winner:
        print("el ganador es", winner.name)