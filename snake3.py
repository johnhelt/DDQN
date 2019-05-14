#import random
import curses
import keyboard 
import numpy as np


class snakegame():
    def __init__(self,gui=True,seed=None):
        #np.random.seed(seed)
        self.sizex = 10
        self.sizey = 10
        self.xspan = np.arange(1,self.sizex-1)
        self.yspan = np.arange(1,self.sizey-1)
        self.playingfield = [[(i,j) for j in self.yspan] for i in self.xspan]
        self.up = [-1,0]
        self.down = [1,0]
        self.left = [0,-1]
        self.right = [0,1]
        self.done = False
        self.score = 0      
        self.fitness = 0 
        self.fruit = [] 
        self.init_snake()
        self.init_fruit() 
        self.gui = gui
        self.key = []
        self.step = 0
        self.lastfruit = 0
        self.last_distance_to_fruit = 10000
        
        
        if self.gui:       
            self.render_init()
        self.direction = self.right
    def get_validfield(self):
        field = []
        freefield = np.zeros([self.sizex,self.sizey])
        snakeheadfield = freefield
        for row in self.playingfield:
            for element in row:
                [x,y] = (element[0],element[1])
                if [x,y] not in self.snake:
                    freefield[x,y] = 1                    
                    field.append([x,y])
                
                if [x,y] in [self.snakehead]:
                    snakeheadfield[x,y] = 1

    
                    
        return field,freefield, snakeheadfield
    def init_snake(self):
        self.snake = [[np.random.choice(self.xspan),np.random.choice(self.yspan)]]
        self.snakehead = self.snake[0]
    def init_fruit(self):        
        field,_,_=self.get_validfield()
        #print(field)
        if len(field)==0:
            return -1        
        fruitix = np.random.choice(np.arange(0,len(field)))        
        self.fruit = field[fruitix]
        #print(self.fruit,self.snake)
        return 0

    def get_fitness(self):
        return self.fitness
    def get_eyes_vector(self,direction):
        #direction=self.direction
        if direction == self.down: # down
            right = self.left
            left = self.right
        if direction == self.up: # up
            right = self.right
            left = self.left
        if direction == self.left:
            right = self.up
            left = self.down
        if direction == self.right:
            right = self.down
            left = self.up
        return [left,right]

    def get_distance_to_fruit(self,snakehead):        
        distance_to_fruit = np.sqrt((self.fruit[0]-snakehead[0])**2+(self.fruit[1]-snakehead[1])**2)
        return distance_to_fruit
    def get_inputs_new(self):
        _,freefield,_=self.get_validfield()
        states=np.concatenate([freefield.ravel(),np.array([self.direction[0],self.direction[1],self.fruit[0],self.fruit[1],self.snakehead[0],self.snakehead[1]])])[np.newaxis]
        return states
    
    def get_inputs(self):
        
              
      
        _,freefield,snakeheadfield=self.get_validfield()
        freefield=freefield.ravel()
        fraction_occupied = 1-np.sum(freefield)/len(freefield)
        fruit_in_front = 0
        fruit_to_the_left = 0
    
        current_direction = self.direction
        snakehead = self.snakehead

        # Get direction to fruit

        if current_direction == self.up:            
            if self.fruit[0] < snakehead[0]:
               fruit_in_front = 1
            if self.fruit[1] < snakehead[1]:                
                fruit_to_the_left = 1        
        if current_direction == self.down:            
            if self.fruit[0] > snakehead[0]:
                fruit_in_front = 1
            if self.fruit[1] > snakehead[1]:
                fruit_to_the_left = 1
        if current_direction == self.right:            
            if self.fruit[1] > snakehead[1]:
                fruit_in_front = 1
            if self.fruit[0] < snakehead[0]:
                fruit_to_the_left = 1
        if current_direction == self.left:            
            if self.fruit[1] < snakehead[1]:
                fruit_in_front = 1
            if self.fruit[0] > snakehead[0]:
                fruit_to_the_left = 1      
      
        [left,right] = self.get_eyes_vector(self.direction)
        collision_front = self.detect_collision(self.direction)
        collision_left = self.detect_collision(left)
        collision_right = self.detect_collision(right)
                

        # inputs: Direction x, direction y, fraction of field occupied by walls, fruit direction front, fruit direction left, collision left, collision right,
        # (1,100) vector with all free fields (1) and occupied fields (0), (1,100) vector with location of snakehead (not sure this is needed, but I added it anyways)
        inputs = np.array([self.direction[0],self.direction[1],fraction_occupied,fruit_in_front,fruit_to_the_left,collision_front,collision_left,collision_right] + list(freefield)+ list(snakeheadfield.ravel()))[np.newaxis]
        return inputs

    def update_snake(self,direction):
        #print(self.snake,self.snake[0])
                  
        new_snake_head_x =self.snake[0][0]+direction[0]
        new_snake_head_y = self.snake[0][1] +direction[1]
        new_snake_head = [new_snake_head_x,new_snake_head_y]
        self.snakehead = new_snake_head
        if self.fruit_eaten() == 1:            
            self.snake = [new_snake_head]+self.snake                        
        else:
            self.snake = [new_snake_head]+self.snake[:-1] 
        self.direction = direction
        #print(len(self.snake))
        #print(self.snake)           
    def fruit_eaten(self):
        fruit_eaten = 0
        if self.snakehead == self.fruit:
            fruit_eaten = 1
                     
        #print("fruit: " + str(self.fruit) + " snakehead: " + str(self.snake[0]))
        return fruit_eaten
        

    def detect_collision(self,direction):
        collision = 0
        new_snake_head_x = self.snake[0][0]+direction[0]
        new_snake_head_y = self.snake[0][1]+direction[1]
        new_snake_head = [new_snake_head_x,new_snake_head_y]
        if new_snake_head[0] == 0:
            collision = 1
        elif new_snake_head[0] == self.sizex+1:
            collision = 1
        elif new_snake_head[1] == 0:
            collision = 1
        elif new_snake_head[1] == self.sizey+1:
            collision = 1
        elif new_snake_head in self.snake:
            collision = 1
        
        return collision
        
    def render_init(self):
        stdscr=curses.initscr()
        
        win = stdscr.derwin(self.sizex + 2, self.sizey+ 2, 0, 0)
        curses.curs_set(0)
        win.nodelay(1)
        win.timeout(1)
        self.win = win
        
        self.render()

    def render(self):
        self.win.clear()
        self.win.border(0)
        self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
        #print(self.fruit,self.fruit[0])
        #self.win.addch(self.food[0], self.food[1], '🍎')
        self.win.addch(self.fruit[0], self.fruit[1], 'A')
        for i, point in enumerate(self.snake):
            if i == 0:
                #self.win.addch(point[0], point[1], '🔸')
                self.win.addch(point[0], point[1], '#')
            else:
                #self.win.addch(point[0], point[1], '🔹')
                self.win.addch(point[0], point[1], '-')
        key=self.win.getch()    
        self.key = key
        
        
    def run_game(self,nn_direction=[]):
        #key = False
        while True:
            status=self.step_game()
            if status < 0:
                break  
            if self.gui == True:
                self.render()
                curses.napms(100)
            
              
            
    def validate_game(self,nn_direction):
        self.render()
        curses.napms(25)
        status=self.step_game(nn_direction)
        return status    

    def step_game(self,nn_direction=[]):
        old_direction = self.direction
        new_direction = old_direction
        

        if nn_direction == 0: # continue straight
            new_direction = old_direction
        if nn_direction == 1: # turn left
            new_direction = [np.cross([1,0],np.array(old_direction)).item(0),np.cross([0,1],np.array(old_direction)).item(0)]
        if nn_direction == 2: # turn right
            new_direction = [np.cross([1,0],-np.array(old_direction)).item(0),np.cross([0,1],-np.array(old_direction)).item(0)]

   
        if (self.key == 119): #or (nn_direction == 0): # up/w
            new_direction = self.up
        if (self.key == 115):# or (nn_direction == 1): #down/s
            new_direction = self.down
        if (self.key == 97):# or (nn_direction == 2): #left/a
            new_direction = self.left
        if (self.key == 100):# or (nn_direction == 3): #right/d
            new_direction = self.right

        if all(np.array(old_direction) == -1*np.array(new_direction)):
            new_direction = old_direction            

        # Collision detected after movement: Game over
        if self.detect_collision(new_direction):
            self.fitness -= 10            
            return -1
        
        # Move snake

        self.update_snake(new_direction)
        # Maximum number of steps w.o. eating fruit reached: Game over
        if self.lastfruit > 100:
            self.fitness -= 10            
            return -2

        # Got to fruit, add point and bonus to fitness
        if self.fruit_eaten():
            self.score = self.score+1            
            self.lastfruit = 0
            self.fitness += 5#20
            distance_to_fruit = 9999
            fruitok=self.init_fruit()
            # Could not place fruit (No more space): Game over
            if fruitok < 0:
                return -3
        else:            
            distance_to_fruit = self.get_distance_to_fruit(self.snakehead)
            if distance_to_fruit < self.last_distance_to_fruit:
                self.fitness += 1
            else:
                self.fitness -= 1.5#1.5
            self.lastfruit += 1

        
        #self.fitness += 1
        self.last_distance_to_fruit = distance_to_fruit
        self.step += 1
        #print(self.fitness)
        return 0
                       
if __name__ == "__main__":
    snake=snakegame(gui=True)
    snake.run_game()

        
