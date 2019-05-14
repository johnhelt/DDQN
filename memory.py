from collections import deque
import numpy as np
import random
class Memory():
    
    def __init__(self,max_memory_size=50000):
        self.max_memory_size = max_memory_size
        self.items = deque(np.zeros(max_memory_size)) 
        self.e = 0.01
        self.a = 0.6

    def binary_search_leftmost(self, x, v): # code from wikipedia...
     
        L = 0
        R = len(v)
    
        while L < R:
            M = int(np.floor((L+R)/2))
           
            if v[M] < x:
                L = M+1                
            else:
                R = M
        
        return L

    def binary_search_rightmost(self, x, v):
     
        L = 0
        R = len(v)
   
        while L < R:
            M = int(np.floor((L+R)/2))
           
            if v[M] <= x:
                L = M+1                
            else:
                R = M

        
        return L-1

    def get(self, lower, upper, acc_p):

        # since acc_p is a sorted (cumsum) array, we can use a binary search algorithm to 
        # locate the interval end points. This is much faster than searching with conditional
        # statements

        lower_ix = self.binary_search_leftmost(lower,acc_p)
        upper_ix = self.binary_search_rightmost(upper,acc_p)
        ix = np.arange(lower_ix,upper_ix)
        
        ix_ = random.choice(ix)                               
        
        memory_ = self.items[ix_]        
        return memory_,ix_        

    def add(self, error, sample):

        priority = (error+self.e)**self.a
        self.items.append((priority, sample))
        if len(self.items) > self.max_memory_size:
            self.items.popleft()
        

    def update(self, errors, ix):

        for error, i in zip(errors, ix):
            priority = (error+self.e)**self.a
            mem = np.array(self.items[i])
            mem[0] = priority
            self.items[i] = tuple(mem)        

    def select_random_batch(self, batch_size): 
        memories = random.sample(list(enumerate(self.items)),batch_size)
        batch = []
        ix = []
        for memory_ in memories:
            ix.append(memory_[0])
            batch.append(memory_[1][1])

        return np.array(batch),ix

    def select_batch(self,batch_size):
        batch = []
        ix = []
        
        priorities = [col[0] for col in self.items]
        ptot = sum(priorities)
        dp = ptot/batch_size

        # sorted array. Following method in deepmind paper.
        acc_p = np.cumsum(priorities)

        for n in range(batch_size):      
            lower = n*dp
            upper = (n+1)*dp      
            
            memory_, ix_ = self.get(lower, upper, acc_p)

            # memory_[0] contains priorities. These are not needed for training, only selection
            batch.append(memory_[1]) 
            ix.append(ix_)
        
        return np.array(batch), ix