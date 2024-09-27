from pokemon import softmax, sample_from_distribution
from tools.pick_six import agent_create_team
from math import isclose as nreq, floor as flr
import google.generativeai as genai
from pkmn_logic_utils import get_U
from sim.player import Decision
import networkx as nx
import sim.sim as sim
import pyreason as pr
from data import dex
import pandas as pd
import random
import time
import json
import re
import os


all_moves = pd.read_json("data/moves.json")#['10000000voltthunderbolt'].
all_abilities = pd.read_json("data/abilities.json")
typechart = pd.read_json("data/typechart.json")
types = list(typechart)
cases = [
    "doesn't affect pokemons with type",
    "is very weak against pokemons with type",
    "is weak against pokemons with type",
    "is normally effective against pokemons with type",
    "is very effective against pokemons with type",
    "is super effective against pokemons with type",
]

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
                self.plan_library[('use', mov)] = Decision('move', i)
        for i, pok in enumerate(px.pokemon):
            if (not pok.fainted) and (not pok.species == px.active_pokemon[0].species):
                self.plan_library[('switch', pok.species)] = Decision('switch', i)
        # print(list(self.plan_library))

    def choose(self, me: sim.Player, foe: sim.Player):
        self.get_available_moves(me)
        try:
            return self.plan_library[random.choice(list(self.plan_library))]
        except:
            return Decision("pass", 0)

class TrainerBDI(TrainerBase):
    def __init__(self, name, beliefs) -> None:
        super().__init__(name)

        self.beliefs = set(beliefs) # agent pr rules ()
        self.desires = {} # plans, with an index of probability
        # i commented the following because the intentions are selected from desires using softmax (1.23), 
        # and we keep track of old desires using a weighted value from previous iterations
        # self.intentions = set() #choosen desires
        self.CONFIDENCE = 2.71
        self.REGRET = 0.1
    
    def choose(self, me: sim.Player, foe: sim.Player):
        # start the inference engine
        pr.reset()
        pr.reset_rules()
        for rule in self.beliefs:
            pr.add_rule(rule)
        
        # print("loading actions")
        # obtain available actions
        self.get_available_moves(me)
        
        # print("loading kgraph")
        # load the basic starting knowledge point
        kg: nx.DiGraph = nx.read_graphml('data/knowledge_graph.graphml')
        kg.add_node("use", is_use = 1)
        kg.add_node("switch", is_switch = 1)
        kg.add_node("pass")
        for can_do in self.plan_library:
            if type(can_do) is tuple:
                u, v = can_do
                kg.add_edge(u, v, can = 1)
            else:
                kg.add_node(can_do, can = 1)

        # add enemy data
        foe_pok = foe.active_pokemon[0]
        kg.add_node("foe")
        pk_hp_p = foe_pok.hp / foe_pok.maxhp
        pk_hp = "low_hp" if pk_hp_p< 0.2 else "med_hp" if pk_hp_p< 0.7 else "high_hp"
        kg.nodes["foe"][pk_hp] = 1

        pk_type = foe_pok.types[0]+'/'+foe_pok.types[1] if len(foe_pok.types)>1 else foe_pok.types[0]
        kg.add_edge("foe", pk_type, pokemon_type = 1)
        for mov in foe_pok.moves:
            kg.add_edge("foe", mov, has_move = 1)
            kg.add_edge(mov, all_moves[mov].type, move_type = 1)
        
        # add self pokemon data
        for pok in me.pokemon:
            kg.add_node(pok.species)
            if pok.fainted:
                kg.nodes[pok.species]['fainted'] = 1
            pk_hp_p = pok.hp / pok.maxhp
            pk_hp = "low_hp" if pk_hp_p< 0.2 else "med_hp" if pk_hp_p< 0.7 else "high_hp"
            kg.nodes[pok.species][pk_hp] = 1
            if pok.species == me.active_pokemon[0].species:
                kg.nodes[pok.species]["on_field"] = 1
                kg.add_edge(pok.species, 'foe', on_field_against = 1)
            else:
                kg.add_edge(pok.species, 'foe', is_foe = 1)
            pk_type = pok.types[0]+'/'+pok.types[1] if len(pok.types)>1 else pok.types[0]
            kg.add_edge(pok.species, pk_type, pokemon_type = 1)
            for mov in pok.moves:
                kg.add_edge(pok.species, mov, has_move = 1)
                kg.add_edge(mov, all_moves[mov].type, move_type = 1)


        # print("inserting kgraph")
        pr.load_graph(kg)
        pr.settings.save_graph_attributes_to_trace = True
        pr.settings.atom_trace=True
        pr.settings.verbose=False
        # print("reasoning")
        interpretation = pr.reason(timesteps=3)
        
        for plan in list(self.desires):
            if plan in self.plan_library:
                self.desires[plan]=self.desires[plan]*self.REGRET
            else:
                self.desires.pop(plan)
    
        ll = pr.filter_and_sort_edges(interpretation, ['desire_ll'])[-1]
        l = pr.filter_and_sort_edges(interpretation, ['desire_l'])[-1]
        m = pr.filter_and_sort_edges(interpretation, ['desire_m'])[-1]
        h = pr.filter_and_sort_edges(interpretation, ['desire_h'])[-1]
        hh = pr.filter_and_sort_edges(interpretation, ['desire_hh'])[-1]
        desires_lists = [ll,l,m,h,hh]
        
        for d_list in desires_lists:
            for i, r in d_list.iterrows():
                if r["component"] in self.desires:
                    self.desires[r["component"]] = self.desires[r["component"]] +self.CONFIDENCE
                else:
                    self.desires[r["component"]] = self.CONFIDENCE
        
        if len(self.desires) == 0:
            self.desires['pass'] = 1
            print(self.name, 'passes his turn')

        x = get_frequencies(self.desires)
        p = softmax(x)
        dec_ind = sample_from_distribution(p)
        choice = get_dict_by_index(self.desires, dec_ind)
        ret = self.plan_library[choice]

        return ret
    
class TrainerBDIF(TrainerBDI): # BDI trainer with less knowledge graph size
    def __init__(self, name, beliefs) -> None:
        super().__init__(name, beliefs)
    
    def choose(self, me: sim.Player, foe: sim.Player):
        # start the inference engine
        pr.reset()
        pr.reset_rules()
        for rule in self.beliefs:
            pr.add_rule(rule)
        
        # print("loading actions")
        # obtain available actions
        self.get_available_moves(me)
        
        # print("loading kgraph")
        # load the basic starting knowledge point
        typechart = pd.read_json("data/typechart.json")
        types = list(typechart)

        kg = nx.DiGraph()
        for pok in foe.pokemon:
            tp1 = pok.types[0]
            tp2 = tp1 if len(pok.types)<=1 else pok.types[1]
            for attack_type in types:
                if tp1 == tp2:
                    eff = float(typechart[tp1]["damage_taken"][attack_type])
                    kg.add_edge(attack_type, tp1, effectiveness = eff/4)
                else:
                    eff = float(typechart[tp1]["damage_taken"][attack_type])*float(typechart[tp2]["damage_taken"][attack_type])
                    kg.add_edge(attack_type, tp1+'/'+tp2, effectiveness = eff/4)

        kg.add_node("use", is_use = 1)
        kg.add_node("switch", is_switch = 1)
        kg.add_node("pass")
        for can_do in self.plan_library:
            if type(can_do) is tuple:
                u, v = can_do
                kg.add_edge(u, v, can = 1)
            else:
                kg.add_node(can_do, can = 1)

        # add enemy data
        foe_pok = foe.active_pokemon[0]
        kg.add_node("foe")
        pk_hp_p = foe_pok.hp / foe_pok.maxhp
        pk_hp = "low_hp" if pk_hp_p< 0.2 else "med_hp" if pk_hp_p< 0.7 else "high_hp"
        kg.nodes["foe"][pk_hp] = 1

        pk_type = foe_pok.types[0]+'/'+foe_pok.types[1] if len(foe_pok.types)>1 else foe_pok.types[0]
        kg.add_edge("foe", pk_type, pokemon_type = 1)
        for mov in foe_pok.moves:
            kg.add_edge("foe", mov, has_move = 1)
            kg.add_edge(mov, all_moves[mov].type, move_type = 1)
        
        # add self pokemon data
        for pok in me.pokemon:
            kg.add_node(pok.species)
            if pok.fainted:
                kg.nodes[pok.species]['fainted'] = 1
            pk_hp_p = pok.hp / pok.maxhp
            pk_hp = "low_hp" if pk_hp_p< 0.2 else "med_hp" if pk_hp_p< 0.7 else "high_hp"
            kg.nodes[pok.species][pk_hp] = 1
            if pok.species == me.active_pokemon[0].species:
                kg.nodes[pok.species]["on_field"] = 1
                kg.add_edge(pok.species, 'foe', on_field_against = 1)
            else:
                kg.add_edge(pok.species, 'foe', is_foe = 1)
            pk_type = pok.types[0]+'/'+pok.types[1] if len(pok.types)>1 else pok.types[0]
            kg.add_edge(pok.species, pk_type, pokemon_type = 1)
            for mov in pok.moves:
                kg.add_edge(pok.species, mov, has_move = 1)
                kg.add_edge(mov, all_moves[mov].type, move_type = 1)


        # print("inserting kgraph")
        pr.load_graph(kg)
        pr.settings.save_graph_attributes_to_trace = True
        pr.settings.atom_trace=True
        pr.settings.verbose=False
        # print("reasoning")
        interpretation = pr.reason(timesteps=3)
        
        for plan in list(self.desires):
            if plan in self.plan_library:
                self.desires[plan]=self.desires[plan]*self.REGRET
            else:
                self.desires.pop(plan)
    
        ll = pr.filter_and_sort_edges(interpretation, ['desire_ll'])[-1]
        l = pr.filter_and_sort_edges(interpretation, ['desire_l'])[-1]
        m = pr.filter_and_sort_edges(interpretation, ['desire_m'])[-1]
        h = pr.filter_and_sort_edges(interpretation, ['desire_h'])[-1]
        hh = pr.filter_and_sort_edges(interpretation, ['desire_hh'])[-1]
        desires_lists = [ll,l,m,h,hh]
        
        for d_list in desires_lists:
            for i, r in d_list.iterrows():
                if r["component"] in self.desires:
                    self.desires[r["component"]] = self.desires[r["component"]] +self.CONFIDENCE
                else:
                    self.desires[r["component"]] = self.CONFIDENCE
        
        if len(self.desires) == 0:
            # print(self.name, 'choosed randomly')
            return self.plan_library[random.choice(list(self.plan_library))]

        x = get_frequencies(self.desires)
        p = softmax(x)
        dec_ind = sample_from_distribution(p)
        choice = get_dict_by_index(self.desires, dec_ind)
        ret = self.plan_library[choice]

        return ret
    
def get_frequencies(dct):
    return [dct[x] for x in list(dct)]
def get_dict_by_index(dct, i):
    return list(dct)[i]

class TrainerLLM(TrainerBase):
    def __init__(self, name, talkative = False, specific_instructions = "") -> None:
        super().__init__(name)
        self.model = None
        self.started = False
        self.speak = talkative
        self.specific_instructions = specific_instructions

    def initialize_model(self, me: sim.Player, foe: sim.Player, inst):
        system_inst = "You are an expert pokemon trainer with a lot of data about pokemon following this line.\n\n"
        system_inst = system_inst + "Data about typecharts for the combat:\n"
        for pok in foe.pokemon + me.pokemon:
            tp1 = pok.types[0]
            tp2 = tp1 if len(pok.types)<=1 else pok.types[1]
            for attack_type in types:
                if tp1 == tp2:
                    eff = float(typechart[tp1]["damage_taken"][attack_type])
                    for x in range(6):
                        if nreq(flr(2**(x-1))*(1/4),eff):
                            system_inst = system_inst + "Type '"+attack_type+"' "+cases[x]+" '"+tp1+"'\n"
                else:
                    eff = float(typechart[tp1]["damage_taken"][attack_type])*float(typechart[tp2]["damage_taken"][attack_type])
                    for x in range(6):
                        if nreq(flr(2**(x-1))*(1/4),eff):
                            system_inst = system_inst + "Type '"+attack_type+"' "+cases[x]+"s '"+tp1+"'"+"and '"+tp2+"'\n"
        system_inst = system_inst + "\n\n"
        system_inst = system_inst + "Your team data:\n"
        for pok in me.pokemon:
            system_inst = system_inst + "'" + pok.species + "' " + str(pok.types) +"\n"
            system_inst = system_inst + f"ability: {pok.ability}, {all_abilities[pok.ability].shortDesc}\n"
            system_inst = system_inst + "moves:\n"
            for mov in pok.moves:
                system_inst = system_inst + "- " + mov + " (" +all_moves[mov].type + "): " + all_moves[mov].desc +"\n"
            system_inst = system_inst + "\n"
        system_inst = system_inst + f"""
Using this information, each turn the user will give you information about the state of the combat, and a list of available actions,
so you are going to make a reasoning process to make a good choice, and you are going to put an end line telling what the move is going to be.
You will explain your reasoning step by step.

The last line must start with 'INSTRUCTION:'
Examples(each one after the reasoning process):
INSTRUCTION: switch to pikachu
INSTRUCTION: switch to riolu
INSTRUCTION: use thunderbolt
INSTRUCTION: pass the turn

Your name is {self.name} and you will follow the rules listed below:
{inst}

"""

        # print(system_inst)
        # print("\n\nstarted")
        genai.configure(api_key=os.environ["API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_inst)
        self.started = True

    def get_prompt(self, m: sim.Pokemon, f: sim.Pokemon):
        prompt = """State of the battle:
"""
        for nm, pk in zip(["Your", "Foe's"], [m, f]):
            hp_p = pk.hp/ pk.maxhp
            hp_f = str(pk.hp)+"/"+str(pk.maxhp)
            prompt = prompt + f"""{nm} {pk.species} {"is fainted" if hp_p<=0 else "has very low hp" if hp_p <=0.2 else "has low hp" if hp_p<= 0.5 else "has relatively high" if hp_p <= 0.8 else "has perfect hp"} ({hp_f})
"""
        return prompt

    def choose(self, me: sim.Player, foe: sim.Player):
        self.get_available_moves(me)
        # initialize model if still off
        if not self.started:
            # print("starting")
            self.initialize_model(me, foe, self.specific_instructions)
        
        
        prompt = self.get_prompt(me.active_pokemon[0], foe.active_pokemon[0])

        prompt = prompt +"""

Information of your team:
"""
        for pk in me.pokemon:
            hp_p = pk.hp/ pk.maxhp
            hp_f = str(pk.hp)+"/"+str(pk.maxhp)
            prompt = prompt + f"""- {pk.species} {"(fainted)" if hp_p<=0 else "(very low hp)" if hp_p <=0.2 else "(low hp)" if hp_p<= 0.5 else "(high hp)" if hp_p <= 0.8 else "(perfect state)"} [{hp_f}]
"""

        prompt = prompt + "\nAvailable actions:\n"
        for plan in self.plan_library:
            if plan == "pass":
                pass
                # prompt = prompt + "- " +"pass turn\n"
            elif plan[0] == "switch":
                prompt = prompt + "- switch to " + plan[1] + "\n"
            else:
                prompt = prompt + "- use " + plan[1] + "\n"

        print(prompt)
        
        num_tries = 0
        while True:
            if num_tries > 23:
                break
            try:
                num_tries = num_tries +1
                # get an answer code
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=1025,
                        temperature=0.2,
        ),
    ).text           # get plan from nl text
                if self.speak:
                    print(self.name, "'s INNER DIALOGUE:")
                    print(response)
                for plan in self.plan_library:
                    if type(plan) is tuple:
                        if re.search(fr"INSTRUCTION:(\w|\s)*{plan[0]}(\w|\s)*{plan[1]}", response):
                            # print("found answer at try",num_tries)
                            return (self.plan_library[plan])
                    else:
                        if re.search(fr"INSTRUCTION:(\w|\s)*{plan}", response):
                            # print("passed at try",num_tries)
                            return (self.plan_library[plan])
    
            except Exception as e:
                print("REASONING FAIL:", e)
                tm_exp = 2**num_tries
                time.sleep(tm_exp)
            
class TrainerLLMFeed(TrainerLLM):
    def __init__(self, name, talkative=False, specific_instructions="", num_turns = 10) -> None:
        super().__init__(name, talkative, specific_instructions)
        self.battle = None
        self.num_turns = num_turns
        self.states = []

    def get_prompt(self, m: sim.Pokemon, f: sim.Pokemon):
        prompt = """State of the battle:
"""
        for nm, pk in zip(["Your", "Foe's"], [m, f]):
            hp_p = pk.hp/ pk.maxhp
            hp_f = str(pk.hp)+"/"+str(pk.maxhp)
            prompt = prompt + f"""{nm} {pk.species} {"is fainted" if hp_p<=0 else "has very low hp" if hp_p <=0.2 else "has low hp" if hp_p<= 0.5 else "has relatively high" if hp_p <= 0.8 else "has perfect hp"} ({hp_f})
"""     
        curr_logs = self.battle.logs[-(self.num_turns-1):]
        self.states.append(prompt)
        prompt = "Turn historial (includes last turn without last turn actions which you are selecting now....):\n"
        for i, prm in zip(range(self.num_turns), self.states[-self.num_turns:]):
            prompt = prompt + f"#### {len(self.states[-self.num_turns:]) - i} turns ago:\n"
            prompt = prompt + f"{prm}\n"
            try:
                curr_logs[i]
                prompt = prompt + f"Turn Actions:\n"
                for x in curr_logs[i]:
                    prompt = prompt + f"{x}\n"
            except:
                pass
                # print("index error surely")
                
            prompt = prompt + f"\n"
        # print("PROMPT", prompt, sep="\n")
        return prompt

class TrainerLLMFeedSC(TrainerLLMFeed):
    def __init__(self, name, talkative=False, specific_instructions="", num_turns=10, num_sc = 3) -> None:
        super().__init__(name, talkative, specific_instructions, num_turns)
        self.num_sc = num_sc

    def choose(self, me: sim.Player, foe: sim.Player):
        self.get_available_moves(me)
        # initialize model if still off
        if not self.started:
            # print("starting")
            self.initialize_model(me, foe, self.specific_instructions)
        
        
        prompt = self.get_prompt(me.active_pokemon[0], foe.active_pokemon[0])
        prompt = prompt + "\nAvailable actions:\n"
        for plan in self.plan_library:
            if plan == "pass":
                pass
                # prompt = prompt + "- " +"pass turn\n"
            elif plan[0] == "switch":
                prompt = prompt + "- switch to " + plan[1] + "\n"
            else:
                prompt = prompt + "- use " + plan[1] + "\n"



        num_tries = 0
        while True:
            if num_tries > 23:
                break
            try:
                num_tries = num_tries +1
                # get an answer code
                reasonings = []
                for i in range(self.num_sc):
                    reasonings.append(self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=1025,
                        temperature=0.2,
        ),
    ).text)
                #adding each reasoning here
                prompt = prompt + f"\n\nBetween '<>' are some reasonings you went through, analize them in order to get an answer:\n"
                for reasoning in reasonings:
                    prompt = prompt + f"<{reasoning}>\n"
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=1025,
                        temperature=0.2,
        ),
    ).text           # get plan from nl text
                if self.speak:
                    print(self.name, "'s INNER DIALOGUE:")
                    print(response)
                for plan in self.plan_library:
                    if type(plan) is tuple:
                        if re.search(fr"INSTRUCTION:(\w|\s)*{plan[0]}(\w|\s)*{plan[1]}", response):
                            # print("found answer at try",num_tries)
                            return (self.plan_library[plan])
                    else:
                        if re.search(fr"INSTRUCTION:(\w|\s)*{plan}", response):
                            # print("passed at try",num_tries)
                            return (self.plan_library[plan])
    
            except Exception as e:
                print("REASONING FAIL:", e)
                tm_exp = 2**num_tries
                time.sleep(tm_exp)

def pokemon_log_assoc(B: sim.Battle):
    for pok in B.p1.pokemon:
        pok.log = B.log
        pok.owner = B.p1.name
    for pok in B.p2.pokemon:
        pok.log = B.log
        pok.owner = B.p2.name

def do_battle(t1: TrainerBase, t2: TrainerBase, debug = False):
    if debug:
        print("BATTLE STARTED")
        ans = input("would you like to pass turns manually?\n")
        pt = (ans == "y")
            
    B = sim.Battle('single', t1.name, t1.team, t2.name, t2.team)
    t1.battle = B
    t2.battle = B
    MAX_TURNS = 500
    if debug:
        print(*["\n" + B.p1.name + " vs " + B.p2.name], sep = "\n", end= "\n\n")
        print(*[B.p1.name + " pokemons are:"]+[pok.species for pok in B.p1.pokemon]+['\n'], sep = "\n", end= "\n\n")
        print(*[B.p2.name + " pokemons are:"]+[pok.species for pok in B.p2.pokemon]+['\n'], sep = "\n", end= "\n\n")

    while not B.ended and B.turn<MAX_TURNS:
        # this carries log for fainted pokemons
        pokemon_log_assoc(B)
        
        # the players select their next move
        if debug:
            print(t1.name,'is thinking...')
        B.p1.choice = t1.choose(B.p1, B.p2)
        if debug:
            print(t2.name,'is thinking...')
            print()
        B.p2.choice = t2.choose(B.p2, B.p1)

        # do a turn
        sim.do_turn(B)
        if debug:
            for log in B.logs[-1]:
                print(log)
            if pt: input("press enter to continue...")
        # break
    
    if B.winner == 'p1':
        winner = t1
    elif B.winner == 'p2':
        winner = t2
    else:
        winner = None
    if winner:
        B.logs.append(["The Winner is: " + winner.name + "\n"])
    else:
        B.logs.append(["There is no winner\n"])
    
    if debug:
        print(B.logs[-1][0])
    return B.logs, winner




if __name__ == "__main__":
    LESS_SWITCH_RULES=[
        # reglas para la relevancia de deseos
        pr.Rule('desire_h(A,B) <-0 desire_hh(A,B)', 'desire_h_1'),
        pr.Rule('desire_m(A,B) <-0 desire_h(A,B)', 'desire_h_2'),
        pr.Rule('desire_l(A,B) <-0 desire_m(A,B)', 'desire_h_3'),
        pr.Rule('desire_ll(A,B) <-0 desire_l(A,B)', 'desire_h_4'),
        pr.Rule('desire_m(switch,P) <-0 is_switch(switch), is_foe(P, F), on_field_against(C, F), fainted(C)', 'switch_f'),
        
        # reglas para la seleccion de atacar segun efectividad
        pr.Rule('desire_l(move,M) <-0 has_move(P,M), on_field_against(P, F), is_use(move), move_type(M, MT), pokemon_type(F, FT) , effectiveness(MT,FT):[0.2,1.0]', 'move1'),
        pr.Rule('desire_m(move,M) <-0 has_move(P,M), on_field_against(P, F), is_use(move), move_type(M, MT), pokemon_type(F, FT) , effectiveness(MT,FT):[0.4,1.0]', 'move2'),
        pr.Rule('desire_hh(move,M) <-0 has_move(P,M), on_field_against(P, F), is_use(move), move_type(M, MT), pokemon_type(F, FT) , effectiveness(MT,FT):[0.8,1.0]', 'move3'),

        # reglas para el switcheo a un pokemon con mayor ventaja
        pr.Rule('desire_ll(switch,P) <-0 is_foe(P, F), is_switch(switch), has_move(P, M), move_type(M, MT), pokemon_type(F, FT), effectiveness(MT,FT):[0.4,1.0]', 'switch2'),
        pr.Rule('desire_l(switch,P) <-0 is_foe(P, F), is_switch(switch), has_move(P, M), move_type(M, MT), pokemon_type(F, FT), effectiveness(MT,FT):[0.8,1.0]', 'switch3'),
    ]


    # player1 = TrainerBDIF("BDIF", BASELINE_RULES)#, LESS_SWITCH_RULES)
    player1 = TrainerBDIF("LESS_SWITCH", LESS_SWITCH_RULES)
    player2 = TrainerBase("BASE")

    logs, _ = do_battle(player1, player2, debug=True)

    system_inst = "You are a Pokemon Battle Analizer, given a battle Turn sucesion, you are going to give a full narrative of the battle with a general outline of each participant."
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_inst)

    prompt = "The battle went as follows:\n"
    for log in logs:
        for caption in log:
            prompt = prompt+caption+"\n"

    print("prompt")
    print(prompt)
    narrative = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=2048,
                        temperature=0.2,
        ),
    ).text
    print()
    print(narrative)


    
    
    # victor = []
    # for i in range(30):
    #     player1 = TrainerBDIF("BDIF", BASELINE_RULES)#, LESS_SWITCH_RULES)
    #     player2 = TrainerLLM("LLM", talkative= True)

    #     winner, logs = do_battle(player1, player2, debug=True)

    #     # for log in logs:
    #     #     for caption in log:
    #     #         print(caption)
    #     if winner:
    #         # print("el ganador es", winner.name)
    #         victor.append(winner.name)
    # print()
    # print()
    # print("resultados:")
    # print(victor, sep='\n')
    # print()
    # print((victor.count("LLM"))/len(victor))
