from pokemon import Pokemon
from picksix import generateTeam
from ai import Ai

class Side:
    def __init__(self, battle, num, ai=True):
        if ai == True:
            self.ai = Ai(num)
        self.num = num

        if num == 0:
            self.name = 'nick'
        if num == 1:
            self.name = 'sam'

        self.battle = battle

        self.pokemon = []

        self.team = []

        #for i in range(len(self.team)):
        #    self.pokemon.append(Pokemon(self.team[i], num, self, battle))
        #    self.pokemon[i].position = i
        #print(self.pokemon)

        self.pokemonLeft = len(self.pokemon)

        self.activePokemon = None

        '''
            current request is one of
            move - move request
            switch - fainted pokemon or for u-turn and what not
            teampreview - beginning of battle pick which pokemon
            '' - no request
        '''
        self.request = ''

    def populate_team(self, team):
        if team == None:
            self.team = generateTeam()
        else:
            self.team = team

        for i in range(len(self.team)):
            self.pokemon.append(Pokemon(self.team[i], self.num, self, self.battle))
            self.pokemon[i].position = i

        self.pokemonLeft = len(self.pokemon)
        self.activePokemon = self.pokemon[0]
        self.pokemon[0].active = True

    #handle a switch
    def switch(self):
        #default just pick the next pokemon that isnt fainted
        if self.choice == None:
            for pokemon in self.pokemon:
                if pokemon.fainted == False:
                    self.activePokemon = pokemon
                    pokemon.active = True
        #follow the choice of the player
        else:
            if self.pokemon[self.choice.selection].fainted == False:
                self.activePokemon = self.pokemon[self.choice.selection]
                self.pokemon[self.choice.selection].active = True

            

