from data import dex
import sim.sim as sim
import sim.player as player
import pandas as pd
import numpy as np
import random
from sim.structs import Pokemon
from pokemon import PokemonEntity

TURN_BONUS = 10

def get_U():
    return np.random.uniform()

class PokemonDataDx:
    def __init__(self):
        self.index_to_id_pokemon = dex.index_to_id_pokemon
        self.id_to_index_pokemon = {v: k for k, v in dex.index_to_id_pokemon.items()}
    
class ExperienceHandler:
    def __init__(self):
        
        self.pokemon_index_to_xp_data = {} # dict of index to dicts holding xp data by pokemon

        pokemon_exp_yield = (
            pd.read_csv("data/pokemon_ev_yields_ok.csv")[["Number", "Exp. Yield"]]
        )[:849]
        for row in pokemon_exp_yield.iterrows():
            if not int(row[1]["Number"]) in self.pokemon_index_to_xp_data:
                self.pokemon_index_to_xp_data[int(row[1]["Number"])] = {}
                (self.pokemon_index_to_xp_data[int(row[1]["Number"])])["xp_base"] = float(
                    row[1]["Exp. Yield"]
                )

        pokemon_exp_type = (
            pd.read_csv("data/pokemon_experience_types.csv")[["Number", "Experience Type"]]
        )[:807]

        for row in pokemon_exp_type.iterrows():
            if int(row[1]["Number"]) in self.pokemon_index_to_xp_data:
                (self.pokemon_index_to_xp_data[int(row[1]["Number"])])["xp_type"] = str(
                    row[1]["Experience Type"]
                )

        def toin(x):
            xn = str.replace(x,',','')
            return int(xn)

        pokemon_exp_to_level_up = (
            pd.read_csv("data/experience_table.csv")[['Erratic', 'Fast', 'Medium Fast', 'Medium Slow', 'Slow',
            'Fluctuating']]
        )
        self.pokemon_exp_to_level_up_by_type = {} # this holds the leveling requirements for each experience type
        for column in pokemon_exp_to_level_up.columns:
            self.pokemon_exp_to_level_up_by_type[column] = [x for x in map(toin, list(pokemon_exp_to_level_up[column]))]

    def experience_gain(self, winner_lvl, loser_lvl, loser_index):
        b = self.pokemon_index_to_xp_data[loser_index]["xp_base"]
        l = (b * loser_lvl)/5
        r = ((2*loser_lvl+10)/(loser_lvl + winner_lvl + 10))**2.5
        x = l * r + 1
        return int(x)
    
    def leveling_handler(self, pkmn_id, pkmn_lvl, pkmn_total_xp):
        pkmn_ind = PokemonDataDx().id_to_index_pokemon[pkmn_id]
        cum_lvl = self.pokemon_exp_to_level_up_by_type[self.pokemon_index_to_xp_data[pkmn_ind]["xp_type"]]
        for lvl in range(pkmn_lvl,100):
            if pkmn_total_xp >= cum_lvl[lvl]:
                pkmn_lvl = pkmn_lvl + 1
                # print("leveled up to level:",pkmn_lvl)
            else:
                break
        return pkmn_lvl
    
    def base_stats(self, pkmn: Pokemon):
        a = pkmn.stats.hp
        b = pkmn.stats.attack
        c = pkmn.stats.defense
        d = pkmn.stats.specialattack
        e = pkmn.stats.specialdefense
        f = pkmn.stats.speed
        return (a + b + c + d + e + f)/6
    
    def stats_mod(self, pkmn1: Pokemon, pkmn2: Pokemon):
        a = self.base_stats(pkmn1)
        b = self.base_stats(pkmn2)
        x = b/a
        return (((np.tanh(13*(x-1.1))+1)/2)+0.7*x)*1.5**x
    
    
class BasicPkmnLogic:
    def __init__(self) -> None:
        self.experience_handler = ExperienceHandler()
        self.ecosystem = PokemonDataDx()
        self.lb_log = None
        
        self.manual_battle = None
        self.manual_battle_exp = False
        
    def try_run(self, pkmn1: Pokemon, pkmn2: Pokemon):
        U = get_U()
        r = pkmn1.stats.speed/pkmn2.stats.speed
        p = (np.tanh(3*(r - 0.95))+1)/2
        return U < p
            
    def manual_battle_create(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity):
        self.manual_battle_exp = True
        self.manual_battle = sim.Battle('single', 'Piad', pkmn1.entity, 'Daniel', pkmn2.entity)
        
    def manual_battle_do_turn(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity, my_choice: int):
        # print(my_choice)
        pkmn1_choice = my_choice
        pkmn2_choice = pkmn2.ask_for_choice(self.manual_battle.p2.active_pokemon[0], self.manual_battle.p1.active_pokemon[0])
        if not (pkmn1_choice < 4) and not (pkmn2_choice < 4): #both run
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" runs")
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" runs")
            self.manual_battle.log.append("both pokemons runs so the self.manual_battle ends")
            self.manual_battle_exp = False
            return True
        elif not (pkmn1_choice < 4): #pkmn1 run
            success_run = self.try_run(self.manual_battle.p1.active_pokemon[0],self.manual_battle.p2.active_pokemon[0])
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" prepares to run")
            if success_run:
                self.manual_battle.log.append("succesfully")
                self.manual_battle.winner = 'p2'
                self.manual_battle_exp = False
                return True
            else:
                self.manual_battle.log.append("but failed")
                self.manual_battle.p1.choice = player.Decision('pass', 0)
                self.manual_battle.p2.choice = player.Decision('move', pkmn2_choice)
        elif not (pkmn2_choice < 4): #pkmn2 run
            success_run = self.try_run(self.manual_battle.p2.active_pokemon[0],self.manual_battle.p1.active_pokemon[0])
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" prepares to run")
            if success_run:
                self.manual_battle.log.append("succesfully")
                self.manual_battle.winner = 'p1'
                self.manual_battle_exp = False
                return True
            else:
                self.manual_battle.log.append("but failed")
                self.manual_battle.p1.choice = player.Decision('move', pkmn1_choice)
                self.manual_battle.p2.choice = player.Decision('pass', 0)            
        else:
            self.manual_battle.p1.choice = player.Decision('move', pkmn1_choice)
            self.manual_battle.p2.choice = player.Decision('move', pkmn2_choice)
        
        if pkmn1_choice == -1:
            self.manual_battle.p1.choice = player.Decision('pass', 0)
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" passes his turn")
        if pkmn2_choice == -1:
            self.manual_battle.p2.choice = player.Decision('pass', 0)
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" passes his turn")

        sim.do_turn(self.manual_battle)
        
        if self.manual_battle.p1.active_pokemon[0].hp <=0:
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" has fainted")
            return True
        elif self.manual_battle.p2.active_pokemon[0].hp <=0:
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" has fainted")
            return True
        
        self.manual_battle.log.append(f"status: {pkmn1.entity[0].species} hp:{self.manual_battle.p1.active_pokemon[0].hp}/{self.manual_battle.p1.active_pokemon[0].maxhp} and {pkmn2.entity[0].species} hp:{self.manual_battle.p2.active_pokemon[0].hp}/{self.manual_battle.p2.active_pokemon[0].maxhp}")
        if self.manual_battle.ended:
       
            self.lb_log = self.manual_battle.log
            
            if self.manual_battle_exp:
                if self.manual_battle.winner == 'p1':
                    winner = pkmn1
                    loser = pkmn2
                if self.manual_battle.winner == 'p2':
                    winner = pkmn2
                    loser = pkmn1
                loser_index = self.ecosystem.id_to_index_pokemon[loser.entity[0].species]
                modifier = (self.manual_battle.turn+TURN_BONUS)/self.manual_battle.turn
                
                winner.xp = winner.xp + int(modifier*self.experience_handler.experience_gain(winner.lvl, loser.lvl, loser_index))
                winner.lvl = self.experience_handler.leveling_handler(winner.entity[0].species, winner.lvl, winner.xp)
                winner.entity[0].level = winner.lvl
                return True
            
    def battle(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity, debug = False):
        exp = True
        battle = sim.Battle('single', 'A', pkmn1.entity, 'B', pkmn2.entity, debug=debug)  
        MAX_TURNS = 500
        
        while not battle.ended:
            if battle.turn > 500:
                exp = False
                break
            battle.log.append(f"status: {pkmn1.entity[0].species} hp:{battle.p1.active_pokemon[0].hp}/{battle.p1.active_pokemon[0].maxhp} and {pkmn2.entity[0].species} hp:{battle.p2.active_pokemon[0].hp}/{battle.p2.active_pokemon[0].maxhp}")
            pkmn1_choice = pkmn1.ask_for_choice(battle.p1.active_pokemon[0], battle.p2.active_pokemon[0])
            pkmn2_choice = pkmn2.ask_for_choice(battle.p2.active_pokemon[0], battle.p1.active_pokemon[0])
            if not (pkmn1_choice < 4) and not (pkmn2_choice < 4): #both run
                battle.log.append(battle.p1.active_pokemon[0].species+" runs")
                battle.log.append(battle.p2.active_pokemon[0].species+" runs")
                battle.log.append("both pokemons runs so the battle ends")
                exp = False
                break
            elif not (pkmn1_choice < 4): #pkmn1 run
                success_run = self.try_run(battle.p1.active_pokemon[0],battle.p2.active_pokemon[0])
                battle.log.append(battle.p1.active_pokemon[0].species+" prepares to run")
                if success_run:
                    battle.log.append("succesfully")
                    battle.winner = 'p2'
                    exp = False
                    break
                else:
                    battle.log.append("but failed")
                    battle.p1.choice = player.Decision('pass', 0)
                    battle.p2.choice = player.Decision('move', pkmn2_choice)
            elif not (pkmn2_choice < 4): #pkmn2 run
                success_run = self.try_run(battle.p2.active_pokemon[0],battle.p1.active_pokemon[0])
                battle.log.append(battle.p2.active_pokemon[0].species+" prepares to run")
                if success_run:
                    battle.log.append("succesfully")
                    battle.winner = 'p1'
                    exp = False
                    break
                else:
                    battle.log.append("but failed")
                    battle.p1.choice = player.Decision('move', pkmn1_choice)
                    battle.p2.choice = player.Decision('pass', 0)            
            else:
                battle.p1.choice = player.Decision('move', pkmn1_choice)
                battle.p2.choice = player.Decision('move', pkmn2_choice)
            
            if pkmn1_choice == -1:
                battle.p1.choice = player.Decision('pass', 0)
                battle.log.append(battle.p1.active_pokemon[0].species+" passes his turn")
            if pkmn2_choice == -1:
                battle.p2.choice = player.Decision('pass', 0)
                battle.log.append(battle.p2.active_pokemon[0].species+" passes his turn")

            if pkmn1_choice in range(4):
                used = pkmn1.entity[0].moves[pkmn1_choice]
                if used in pkmn1.used_moves:
                    pkmn1.used_moves[used] = pkmn1.used_moves[used]+1
                else:
                    pkmn1.used_moves[used] = 1
                
            sim.do_turn(battle)
        if battle.p1.active_pokemon[0].hp <=0:
            battle.log.append(battle.p1.active_pokemon[0].species+" has fainted")
        if battle.p2.active_pokemon[0].hp <=0:
            battle.log.append(battle.p2.active_pokemon[0].species+" has fainted")
        
        self.lb_log = battle.log
        
        if exp:
            if battle.winner == 'p1':
                winner = pkmn1
                loser = pkmn2
            if battle.winner == 'p2':
                winner = pkmn2
                loser = pkmn1
            loser_index = self.ecosystem.id_to_index_pokemon[loser.entity[0].species]
            modifier = (battle.turn+TURN_BONUS)/battle.turn
            
            winner.xp = winner.xp + int(modifier*self.experience_handler.experience_gain(winner.lvl, loser.lvl, loser_index))
            winner.lvl = self.experience_handler.leveling_handler(winner.entity[0].species, winner.lvl, winner.xp)
            winner.entity[0].level = winner.lvl
        if battle.winner == 'p1':
            return 1
        else:
            return 0
        
    def pokemon_match_old(self, pokemon_list):
        random.shuffle(pokemon_list)
        for i in range(0, len(pokemon_list), 2):
            if i + 1 < len(pokemon_list):  # Ensure there is a pair
                pokemon1 = pokemon_list[i]
                pokemon2 = pokemon_list[i + 1]
                self.battle(pokemon1, pokemon2)
                
    def pokemon_match(self, pokemon_list):
        for i in range(len(pokemon_list)):
            for j in range(i+1, len(pokemon_list)):
                pokemon1 = pokemon_list[i]
                pokemon2 = pokemon_list[j]
                self.battle(pokemon1, pokemon2)
    
    def pokemon_bipartite_reg_matches(self, pokemon_list, t):
        random.shuffle(pokemon_list)
        A = pokemon_list[:int(len(pokemon_list)/2)]
        B = pokemon_list[int(len(pokemon_list)/2):]
        for i in range(int(len(pokemon_list)/2)):
            for j in range(t):
                self.battle(A[i], B[np.mod(j, int(len(pokemon_list)/2))])
                

    
if __name__ == "__main__":
    pass