import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from memory import Memory
from snakenetwork import Network
from snake3 import snakegame


   
class RandomAgent():
    def __init__(self, model = Network(), memory = Memory()):
        self.memory = memory
        self.num_actions = model.num_outputs        
        self.name = "RandomAgent"

    def add_transition(self,sample):
        error = abs(sample[3]) 
        self.memory.add(error, sample)

    def get_action(self, _):
        return random.choice(range(0,self.num_actions))

    def do_replay(self):
        return 0

    def set_target_weights(self):
        pass
    

class DDQNAgent():
    def __init__(self, model = Network(), epsilon=100, epsilon_min = 10,\
        stop_explore = 15000, gamma = 0.9,  \
        batch_size = 32, prioritized_replay = True, train_whole_batch = True, \
        memory = Memory()):

        self.memory = memory
       
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.batch_size = batch_size              
        self.model_action = model
        self.model_target = model
        self.set_target_weights()

        self.num_actions = model.num_outputs
        self.num_inputs = model.num_inputs
       
        self.train_whole_batch = train_whole_batch
        self.prioritized_replay = prioritized_replay
        self.reset_counters()
        
        
        self.stop_explore = stop_explore
        self.decay =  np.log(self.epsilon_min)/self.stop_explore    
        self.name = "DDQNAgent_20x20_highMem" 

    def reset_counters(self):
        self.t = 0
        self.num_games = 0
    
    def get_action(self, state):
        # exploration vs exploitation - use neural network when enough observations have been collected
        
        if random.random() < self.epsilon/100:
            action = random.choice(range(0,self.num_actions))
        else:                     
            action = np.argmax(self.model_action.predict(state))
        
        return action
    
    def get_targets(self,batch):

        targets=np.zeros((batch.shape[0], self.num_actions))
        inputs=np.zeros((batch.shape[0], self.num_inputs))
        errors = np.zeros(batch.shape[0])

        #for i,(action, state, state_, reward, status) in enumerate(batch):
        for i in range(batch.shape[0]):
            action = batch[i][0]
            state = batch[i][1]
            state_ = batch[i][2]
            reward = batch[i][3]
            status = batch[i][4]

            inputs[i:i+1] = state
            target = reward
            q_values_old = self.model_action.predict(state)[0]
            q_values_next = self.model_action.predict(state_)[0]
            best_action_next = np.argmax(q_values_next)
            q_target_next = self.model_target.predict(state_)[0]
            
            if status == 0:
                target=reward+self.gamma*q_target_next[best_action_next]

            target_f = q_values_old
            target_f[action] = target
            targets[i:i+1] = target_f
            errors[i:i+1] = abs(q_values_old[action]-target)
            
        return (inputs, targets, errors)

    def add_transition(self, sample):     

        _, _, error = self.get_targets(np.array(sample)[np.newaxis])
            
        self.memory.add(error, sample)           

        self.epsilon = self.epsilon_min + (100 - self.epsilon_min) * np.exp(-self.decay * self.t)

        self.t+=1        

    def train(self, inputs, targets):
        loss_t = 0
        if self.train_whole_batch:
            loss_t=self.model_action.train_on_batch(inputs, targets)                   
        else:
            for input, target in zip(inputs,targets):
                loss_t+=self.model_action.train_on_batch(np.array(input)[np.newaxis], np.array(target)[np.newaxis])    
        return loss_t
    
    def do_replay(self):
        
        if self.prioritized_replay == False:
            batch, ix = self.memory.select_random_batch(self.batch_size)
        else:
            batch, ix = self.memory.select_batch(self.batch_size)            

        inputs, targets, errors = self.get_targets(batch)
        
        self.memory.update(errors, ix)
        
        loss_t = self.train(inputs,targets)
                
        return loss_t
        
    def set_target_weights(self):        
        self.model_target.set_weights(self.model_action.get_weights())


class Trainer():
    def __init__(self, max_steps=30000, instance=0, update_target_network=10000):        
        
        self.max_steps = max_steps        
        self.instance = instance
        self.update_target_network = update_target_network       

    def expavg(self, timeseries, alpha=0.001):
        e = np.zeros(len(timeseries))
        e[0]=timeseries[0]
        
        for k in range(len(timeseries)-1):
            e[k+1] =e[k]*(1-alpha)+alpha*timeseries[k]
        return e

    def train(self, agent, game, max_steps, save_model):
    
                   
        loss = []
        fitnesshistory = []
        steps = 0
        num_games = 0
        replays = 0

        pbar = tqdm(total=max_steps)

        while steps <= max_steps:
            
            game.__init__(gui=False, seed=None)
            
            state = game.get_inputs()
            status = 0
            while status >= 0:
                
                action = agent.get_action(state)                
                fitness = game.get_fitness()
                # Step to next state                                
                status = game.step_game(action)                
                state_ = game.get_inputs()
                fitness_ = game.get_fitness()
                reward = fitness_-fitness
                fitnesshistory.append(fitness)

                agent.add_transition((action, state, state_, reward, status))                                
                
                if steps % self.update_target_network == 0:
                    agent.set_target_weights()
                state = state_ 
                
                if steps % 100 == 0:
                    pbar.update(100)                                
                if status < 0:                    
                    num_games += 1
                                        
                                                                                                        
                if steps % 10 == 0:
                    loss_t = agent.do_replay()
                    loss.append(loss_t)
                    replays += 1
                steps += 1    

                if (save_model == True) and (steps%10000 == 0):
                    agent.model_action.save_model('./bestfit_' + agent.name + '_' + str(self.instance) + '_step_'+str(steps)+ '.h5')                      
                    plt.subplot(2,1,1,title="Reward")
                    #plt.plot(np.arange(0,self.episodes),fitnesshistory)
                    #plt.plot(np.arange(0,self.episodes),self.expavg(fitnesshistory),label='iter_'+str(self.instance))
                    plt.plot(np.arange(0,steps),self.expavg(fitnesshistory),label=agent.name + '_iter_' + str(self.instance) + "_reward")
                    plt.legend()
                    plt.ylim(0,150)
                    plt.xlabel("Steps")
                    plt.ylabel("Averaged reward")
                    plt.subplot(2,1,2,title="Loss")            
                    plt.plot(np.arange(0,replays),self.expavg(loss,alpha=0.01),label=agent.name + '_iter_' + str(self.instance)+ "_loss")
                    plt.xlabel('Replays')
                    plt.ylabel('Averaged loss')
                    plt.savefig(agent.name + '_iter_' + str(self.instance))
                    plt.close()
                    self.instance += 1  
                    
          


        pbar.close()
        print("Steps: " + str(steps) + ". Average fitness:" + str(np.mean(fitnesshistory)) + ". Standard deviation: " + str(np.std(fitnesshistory)))          
        
        return loss_t, steps, num_games    


class Wrapper():
    def __init__(self):
        pass
    def run(self):
# Constants
        game = snakegame(gui=False,seed=1)
        
        MAX_MEMORY_SIZE = 100000
        NEURONS = 6#6
        LAYERS = 2#2
        NUM_INPUTS = 8#208#8
        NUM_OUTPUTS = 3
        LEARNING_RATE = 0.005
        MAX_GAME_STEPS = 300000
        LAMBDA = 0

        # Setup trainer, memory and model architecture

        trainer = Trainer()
        memory = Memory(max_memory_size = MAX_MEMORY_SIZE)
        model = Network(neurons = NEURONS, layers = LAYERS, num_inputs = NUM_INPUTS, num_outputs = NUM_OUTPUTS, learning_rate = LEARNING_RATE, l = LAMBDA)

        # Create random agent to fill memory and train it 
        rand_agent = RandomAgent(memory = memory, model = model)
        trainer.train(rand_agent, game, max_steps = MAX_MEMORY_SIZE, save_model = False)

        # Create smart agent and train it, use memory from random agent
        smart_agent = DDQNAgent(memory = rand_agent.memory, model = model, prioritized_replay=True, epsilon_min = 10, stop_explore = 30000)
        trainer.train(smart_agent, game, max_steps = MAX_GAME_STEPS, save_model = True)
wrap=Wrapper()
wrap.run()