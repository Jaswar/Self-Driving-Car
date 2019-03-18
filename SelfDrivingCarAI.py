# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:46:33 2019

@author: janwa
"""

import numpy as np
import vectormath as vmath
import noise
import random
import pyglet
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

learningRate = 0.00001    #learning rate for the model
batchSize = 64            #size of the batch that neural network is being trained on every iteration
memSize = 50000           #size of deep q learning memory
gamma = 0.9               #gamma that is the value on how much next state influences previous state
nActions = 3              #how many actions Ai can take
epsilon = 1.              #initial epsilon also known as exploration rate
epsilonDecayRate = 0.002  #the amount of how much epsilon is being lowered every epoch
minEpsilon = 0.01         #the lowest possible exploration rate
nEpochs = 30000           #how many epochs should AI train
nInputs = 11              #how many inputs neural network has
saveRate = 300            #every how many epochs a model is saved

trainMode = False         #if we train a model we set it to True if we test it we set it to False
filepathToOpen = 'model.h5'  #filepath to open and test a pretrained model
filepathToSave = 'newModel.h5' #filepath to save a model

def rotate(x, y, radians):
    #This uses numpy to build a rotation matrix and take the dot product
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])

def mapp(value, v_min, v_max, new_min, new_max):
    new_value = (value - v_min)*(new_max - new_min)/(v_max - v_min) + new_min
    return new_value

class Car:
        def __init__(self, x0, y0, max_vel):
            self.x = x0
            self.y = y0
            self.max_vel = max_vel
            self.angle = 0
            self.keys = dict(left=False, right=False, up=True, down=False)
            self.vel = vmath.Vector2(0, 0)
            #self.color = color('black')
            
        def update(self, drag):
            #k is the time set...useless
            self.x += self.vel.x * 3
            self.y += self.vel.y * 3
            self.vel = self.vel.__mul__(drag)
            if self.angle < 0:
                self.angle += np.pi*2
            elif self.angle > np.pi*2:
                self.angle -= np.pi*2

        def move(self, acc, ang_increment):
            #k is the time set...useless
            if self.keys['left']:
                self.angle += ang_increment
            if self.keys['right']:
                self.angle -= ang_increment
            if self.keys['up']:
                self.accelerate(acc)
            if self.keys['down']:
                self.accelerate(-acc)

        def accelerate(self, acc):
            self.vel = self.vel.__add__(vmath.Vector2(acc*np.cos(self.angle), acc*np.sin(self.angle)))
            if self.vel.length > self.max_vel:
                self.vel = self.vel.as_length(self.max_vel)

        def get_lines(self):
            v1 = vmath.Vector2(15, -7.5)
            v1.x, v1.y = rotate(v1.x, v1.y, -self.angle)
            v2 = vmath.Vector2(-15, -7.5)
            v2.x, v2.y = rotate(v2.x, v2.y, -self.angle)
            v3 = vmath.Vector2(15, 7.5)
            v3.x, v3.y = rotate(v3.x, v3.y, -self.angle)
            v4 = vmath.Vector2(-15, 7.5)
            v4.x, v4.y = rotate(v4.x, v4.y, -self.angle)
            return [
            [self.x + v1.x, self.y + v1.y, self.x + v2.x, self.y + v2.y],
            [self.x + v3.x, self.y + v3.y, self.x + v4.x, self.y + v4.y]
            ]

class Game:
    def __init__(self):
        self.reward_amount_backup = 1
        self.reward_amount = self.reward_amount_backup
        self.reward_gradient = 0.9999
        self.height = 800
        self.width = 1000
        self.acceleration = 0.5/2
        self.drag = 0.5**0.5
        self.ang_increment = 0.25/2
        self.view_range = 500
        self.noiseMax = 12/16
        self.circuit_offset = 100
        self.lines_len = 20
        self.circuit_detail = 360/self.lines_len
        self.vision_angles = [-np.pi/12, 0, np.pi/12, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        self.lines = [[0, 0, 0, 0]]
        self.reward_lines = []
        self.reward_lines_backup = []
        #self.lines, self.reward_lines = self.perlinLoopCoords(self.lines) #Use this to generate every time a random circuit, and so remove the next 2 lines, that are the saved circuit, if you like a circuit just get it's env.lines and env.reward_lines and copy them in the 2 lines below!
        self.lines = [[727.8698088267481, 325.9606109353626, 786.4457299311956, 400.0], [786.4457299311956, 400.0, 799.8577893461647, 497.42970182742664], [799.8577893461647, 497.42970182742664, 724.7380561204255, 563.2817554327446], [724.7380561204255, 563.2817554327446, 623.8517245497035, 570.4672744893877], [623.8517245497035, 570.4672744893877, 553.3840429895191, 564.2991902566989], [553.3840429895191, 564.2991902566989, 500.0, 572.5230316321056], [500.0, 572.5230316321056, 440.9587011143799, 581.7104335937167], [440.9587011143799, 581.7104335937167, 371.9884470820967, 576.1927870476411], [371.9884470820967, 576.1927870476411, 314.3491025112757, 534.8832723879219], [314.3491025112757, 534.8832723879219, 296.03193705629866, 466.27324105288176], [296.03193705629866, 466.27324105288176, 297.91177014509833, 400.0], [297.91177014509833, 400.0, 276.8101401870964, 327.4812185473265], [276.8101401870964, 327.4812185473265, 247.81050510778545, 216.77360684461541], [247.81050510778545, 216.77360684461541, 272.74090331796526, 87.20468806423696], [272.74090331796526, 87.20468806423696, 372.03235713650884, 6.156092267911049], [372.03235713650884, 6.156092267911049, 499.99999999999994, 9.327296415964724], [499.99999999999994, 9.327296415964724, 606.0463753775093, 73.62281632353233], [606.0463753775093, 73.62281632353233, 670.1918846463659, 165.75096696182626], [670.1918846463659, 165.75096696182626, 696.6546419136769, 257.1220393200481], [696.6546419136769, 257.1220393200481, 727.8698088267481, 325.9606109353625], [632.7641571972326, 356.86231037285734, 686.4457299311956, 400.0], [686.4457299311956, 400.0, 704.7521377166494, 466.52800238993194], [704.7521377166494, 466.52800238993194, 643.8363566829307, 504.5032302034973], [643.8363566829307, 504.5032302034973, 565.0731993204562, 489.56557505189295], [565.0731993204562, 489.56557505189295, 522.4823435520244, 469.19353862718356], [522.4823435520244, 469.19353862718356, 500.0, 472.5230316321055], [500.0, 472.5230316321055, 471.86040055187465, 486.6047819642013], [471.86040055187465, 486.6047819642013, 430.766972311344, 495.2910876101463], [430.766972311344, 495.2910876101463, 395.25080194877046, 476.10474715867457], [395.25080194877046, 476.10474715867457, 391.137588685814, 435.371541615387], [391.137588685814, 435.371541615387, 397.91177014509833, 400.0], [397.91177014509833, 400.0, 371.9157918166118, 358.3829179848212], [371.9157918166118, 358.3829179848212, 328.7122045452802, 275.5521320738627], [328.7122045452802, 275.5521320738627, 331.51942854721256, 168.10638750173172], [331.51942854721256, 168.10638750173172, 402.9340565740036, 101.26174389742641], [402.9340565740036, 101.26174389742641, 499.99999999999994, 109.32729641596472], [499.99999999999994, 109.32729641596472, 575.1446759400145, 168.72846795304773], [575.1446759400145, 168.72846795304773, 611.4133594171187, 246.65266639932102], [611.4133594171187, 246.65266639932102, 615.7529424761822, 315.90056454929544], [615.7529424761822, 315.90056454929544, 632.7641571972326, 356.8623103728573]]
        self.reward_lines = [[727.8698088267481, 325.9606109353626, 632.7641571972326, 356.86231037285734], [799.8577893461647, 497.42970182742664, 704.7521377166494, 466.52800238993194], [623.8517245497035, 570.4672744893877, 565.0731993204562, 489.56557505189295], [500.0, 572.5230316321056, 500.0, 472.5230316321055], [371.9884470820967, 576.1927870476411, 430.766972311344, 495.2910876101463], [296.03193705629866, 466.27324105288176, 391.137588685814, 435.371541615387], [276.8101401870964, 327.4812185473265, 371.9157918166118, 358.3829179848212], [272.74090331796526, 87.20468806423696, 331.51942854721256, 168.10638750173172], [499.99999999999994, 9.327296415964724, 499.99999999999994, 109.32729641596472], [670.1918846463659, 165.75096696182626, 611.4133594171187, 246.65266639932102]]
        self.reward_lines_backup = self.reward_lines.copy()
        self.car = Car(0, 0, 10/2)
        self.car.x = (self.lines[self.lines_len-1][0] + self.lines[self.lines_len*2-1][0])/2
        self.car.y = (self.lines[self.lines_len-1][1] + self.lines[self.lines_len*2-1][1])/2
        self.car.angle = np.arctan((self.lines[self.lines_len-1][3] - self.lines[self.lines_len-1][1])/(self.lines[self.lines_len-1][2] - self.lines[self.lines_len-1][0]))

    def reset(self):
        self.car.x = (self.lines[self.lines_len-1][0] + self.lines[self.lines_len*2-1][0])/2
        self.car.y = (self.lines[self.lines_len-1][1] + self.lines[self.lines_len*2-1][1])/2
        self.car.angle = np.arctan((self.lines[self.lines_len-1][3] - self.lines[self.lines_len-1][1])/(self.lines[self.lines_len-1][2] - self.lines[self.lines_len-1][0]))
        self.car.vel = vmath.Vector2(0, 0)
        self.reward_lines = self.reward_lines_backup.copy()
        self.reward_amount = self.reward_amount_backup
    
    def perlinLoopCoords(self, lines):
        ran = random.random()*100
        lines1 = [[0, 0, 0, 0]];
        a = np.radians(-self.circuit_detail)
        xoff = mapp(np.cos(a), -1, 1, 0, self.noiseMax) + ran
        yoff = mapp(np.sin(a), -1, 1, 0, self.noiseMax) + ran
        r1 = mapp(noise.snoise2(x = xoff, y = yoff), -1, 1, 100, self.height/1.5)
        x1 = self.width/2 + r1 * np.cos(a)
        y1 = self.height/2 + r1 * np.sin(a)
        if y1 < 0:
            y1 = 0
        elif y1 > self.height:
            y1 = self.height
        lines[len(lines)-1][0] = x1
        lines[len(lines)-1][1] = y1

        r2 = r1 - self.circuit_offset
        x2 = self.width/2 + r2 * np.cos(a)
        y2 = self.height/2 + r2 * np.sin(a)
        if y2 < 0:
            y2 = 0
        elif y2 > self.height:
            y2 = self.height
        lines1[len(lines1)-1][0] = x2
        lines1[len(lines1)-1][1] = y2

        for a in np.arange(0, np.pi*2, np.radians(self.circuit_detail)):
            xoff = mapp(np.cos(a), -1, 1, 0, self.noiseMax) + ran
            yoff = mapp(np.sin(a), -1, 1, 0, self.noiseMax) + ran
            r1 = mapp(noise.snoise2(x = xoff, y = yoff), -1, 1, 100, self.height/1.5)
            x1 = self.width/2 + r1 * np.cos(a)
            y1 = self.height/2 + r1 * np.sin(a)
            if y1 < 0:
                y1 = 0
            elif y1 > self.height:
                y1 = self.height
            lines[len(lines)-1][2] = x1
            lines[len(lines)-1][3] = y1
            lines.append([0, 0, 0, 0]);
            lines[len(lines)-1][0] = x1
            lines[len(lines)-1][1] = y1

            r2 = r1 - self.circuit_offset
            x2 = self.width/2 + r2 * np.cos(a)
            y2 = self.height/2 + r2 * np.sin(a)
            if y2 < 0:
                y2 = 0
            elif y2 > self.height:
                y2 = self.height
            lines1[len(lines1)-1][2] = x2
            lines1[len(lines1)-1][3] = y2
            lines1.append([0, 0, 0, 0])
            lines1[len(lines1)-1][0] = x2
            lines1[len(lines1)-1][1] = y2
        lines.pop()
        lines1.pop()
        reward_lines = []
        for i in np.arange(0, len(lines), 2):
            reward_lines.append([lines[i][0], lines[i][1], lines1[i][0], lines1[i][1]])
        lines.extend(lines1)
        return lines, reward_lines

    def step(self, action):
        self.input(action)
        reward = -0.03;
#        if action == 2:
#            reward = -0.1
        done = False
        self.car.move(self.acceleration, self.ang_increment)
        self.car.update(self.drag)
        color = (150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150)
        c_lines = self.car.get_lines()
        intersect_test = False
        for lin in self.lines:
            if len(lin) == 4:
                
                ta0 = ((lin[1]-lin[3])*(c_lines[0][0]-lin[0]) + (lin[2]-lin[0])*(c_lines[0][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[0][1]-c_lines[0][3]) - (c_lines[0][0]-c_lines[0][2])*(lin[3]-lin[1]))
                tb0 = ((c_lines[0][1]-c_lines[0][3])*(c_lines[0][0]-lin[0]) + (c_lines[0][2]-c_lines[0][0])*(c_lines[0][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[0][1]-c_lines[0][3]) - (c_lines[0][0]-c_lines[0][2])*(lin[3]-lin[1]))

                ta1 = ((lin[1]-lin[3])*(c_lines[1][0]-lin[0]) + (lin[2]-lin[0])*(c_lines[1][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[1][1]-c_lines[1][3]) - (c_lines[1][0]-c_lines[1][2])*(lin[3]-lin[1]))
                tb1 = ((c_lines[1][1]-c_lines[1][3])*(c_lines[1][0]-lin[0]) + (c_lines[1][2]-c_lines[1][0])*(c_lines[1][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[1][1]-c_lines[1][3]) - (c_lines[1][0]-c_lines[1][2])*(lin[3]-lin[1]))
            if (ta0 >= 0 and ta0 <= 1 and tb0 >= 0 and tb0 <= 1) or (ta1 >= 0 and ta1 <= 1 and tb1 >= 0 and tb1 <= 1):
                intersect_test = True

        if intersect_test:
            done = True
            reward = -1

        intersect_test = False
        if len(self.reward_lines) == 0:
            self.reward_lines = self.reward_lines_backup.copy()
        lin = self.reward_lines[0]
        if len(lin) == 4:
            
            ta0 = ((lin[1]-lin[3])*(c_lines[0][0]-lin[0]) + (lin[2]-lin[0])*(c_lines[0][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[0][1]-c_lines[0][3]) - (c_lines[0][0]-c_lines[0][2])*(lin[3]-lin[1]))
            tb0 = ((c_lines[0][1]-c_lines[0][3])*(c_lines[0][0]-lin[0]) + (c_lines[0][2]-c_lines[0][0])*(c_lines[0][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[0][1]-c_lines[0][3]) - (c_lines[0][0]-c_lines[0][2])*(lin[3]-lin[1]))

            ta1 = ((lin[1]-lin[3])*(c_lines[1][0]-lin[0]) + (lin[2]-lin[0])*(c_lines[1][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[1][1]-c_lines[1][3]) - (c_lines[1][0]-c_lines[1][2])*(lin[3]-lin[1]))
            tb1 = ((c_lines[1][1]-c_lines[1][3])*(c_lines[1][0]-lin[0]) + (c_lines[1][2]-c_lines[1][0])*(c_lines[1][1]-lin[1]))/((lin[2]-lin[0])*(c_lines[1][1]-c_lines[1][3]) - (c_lines[1][0]-c_lines[1][2])*(lin[3]-lin[1]))
        if (ta0 >= 0 and ta0 <= 1 and tb0 >= 0 and tb0 <= 1) or (ta1 >= 0 and ta1 <= 1 and tb1 >= 0 and tb1 <= 1):
            intersect_test = lin

        if intersect_test != False:
            self.reward_lines.remove(intersect_test)
            reward = 2
            self.reward_amount = self.reward_amount_backup
            #print(reward, len(self.reward_lines_backup), len(self.reward_lines), self.reward_amount_backup)
        else:
            self.reward_amount *= self.reward_gradient

        distances = []
        for ang in self.vision_angles:
            vect = vmath.Vector2(self.view_range, 0)
            vect.x, vect.y = rotate(vect.x, vect.y, ang - self.car.angle)
            c_line = [self.car.x, self.car.y, self.car.x + vect.x, self.car.y + vect.y]
            min_dist = self.view_range
            for lin in self.lines:
                ta = ((lin[1]-lin[3])*(c_line[0]-lin[0]) + (lin[2]-lin[0])*(c_line[1]-lin[1]))/((lin[2]-lin[0])*(c_line[1]-c_line[3] + 1e-10) - (c_line[0]-c_line[2])*(lin[3]-lin[1]))
                tb = ((c_line[1]-c_line[3])*(c_line[0]-lin[0]) + (c_line[2]-c_line[0])*(c_line[1]-lin[1]))/((lin[2]-lin[0])*(c_line[1]-c_line[3] + 1e-10) - (c_line[0]-c_line[2])*(lin[3]-lin[1]))
                if ta >= 0 and ta <= 1 and tb >= 0 and tb <= 1 and self.view_range*ta < min_dist:
                    min_dist = self.view_range*ta        
            distances.append(min_dist)
        obs = np.zeros((0,0))
        for dist in distances:
           obs = np.append(obs, mapp(dist, 0, self.view_range, 0, 1))
        obs = np.append(obs, mapp(self.car.vel.length, 0, self.car.max_vel, 0, 1))
        obs = np.reshape(obs, (1,11))
        return obs, reward, done

    def input(self, value):
        if value == 0:
            self.car.keys['left'] = True
            self.car.keys['up'] = True
        else:
            self.car.keys['left'] = False
            
        if value == 1:
            self.car.keys['up'] = True
            
        if value == 2:
            self.car.keys['right'] = True
            self.car.keys['up'] = True
        else:
            self.car.keys['right'] = False
            





class Brain(object):
    def __init__(self, nI = 11, nO = 4, lr = 0.001):
        self.model = Sequential()
        self.model.add(Dense(64, activation = 'relu', input_shape = (nI, )))
        self.model.add(Dense(32, activation = 'relu'))
        self.model.add(Dense(nO))
        self.model.compile(optimizer = Adam(lr = lr), loss = 'mse')
        
    def loadModel(self,filepath):
        self.model = load_model(filepath)
        return self.model
  
class DQN(object):
    
    def __init__(self, maxMemory, discount = 0.9):
        self.maxMemory = maxMemory
        self.discount = discount
        self.memory = list()
        
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]
            
    def getBatch(self, model, batchSize):
        lenMemory = len(self.memory)
        numInputs = self.memory[0][0][0].shape[1]
        numOutputs = model.output_shape[-1]
        inputs = np.zeros((min(lenMemory, batchSize), numInputs))
        targets = np.zeros((min(lenMemory, batchSize), numOutputs))
        for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(lenMemory, batchSize))):
            currentState, action, reward, nextState = self.memory[inx][0]
            gameOver = self.memory[inx][1]
            inputs[i] = currentState
            targets[i] = model.predict(currentState)[0]
            Qsa = np.max(model.predict(nextState)[0])
            if gameOver:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Qsa
                
        return inputs, targets  
     
        
    

brain = Brain(nInputs, nActions, learningRate)
model = brain.model
env = Game()
currentState, reward, done = env.step(-1) #IMPORTANT: game_state is made out of 11 inputs, since env.car.angle is gone, and is also a 3D array 12*1*1!!!!
dqn = DQN(memSize, gamma)
if not trainMode:
    model = brain.loadModel(filepathToOpen)
    
def train():
    global currentState
    global epsilon
    
    done = True # env needs to be reset
    
    for epoch in range(nEpochs):
        env.reset()
        game_state, reward, done = env.step(-1)
        currentState = game_state
        totReward = 0.
        loss = 0.
        iteration = 0
        while not done:
            pyglet.clock.tick() #starts to render graphics
            iteration += 1   
            
            action = 0 #as can be seen in the env.input() method, this must be an integer bethween 0 and 3 included
            if trainMode:
                if (np.random.rand() < epsilon):
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(model.predict(currentState)[0])
            else:
                action = np.argmax(model.predict(currentState)[0])
            nextState, reward, done = env.step(action)
            
            if trainMode:
                dqn.remember([currentState, action, reward, nextState], done)
                inputs, targets = dqn.getBatch(model, batchSize)
                loss += model.train_on_batch(inputs, targets)
            
            totReward += reward
            currentState = nextState
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                window.flip()
                
        if epsilon > minEpsilon:
            epsilon -= epsilonDecayRate
        if trainMode and epoch % saveRate == 0:
            model.save(filepathToSave)
        print('Epoch: ' + str(epoch) + ' Epsilon: {:.5f}'.format(epsilon) + ' Average Loss: {:.5f}'.format(loss / iteration) + ' Average reward: {:.3f}'.format(totReward / iteration))

window = pyglet.window.Window(width=1000, height=800)
            
@window.event
def on_draw():
    color = (150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150)
    window.clear()
    for lin in env.lines:
          if len(lin) == 4:
              #line(lin[0], lin[1], lin[2], lin[3]);
              pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                ('v2f', (lin[0], lin[1], lin[2], lin[3])))
    for lin in env.reward_lines:
         if len(lin) == 4:
               #line(lin[0], lin[1], lin[2], lin[3]);
               pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
               ('v2f', (lin[0], lin[1], lin[2], lin[3])), ('c3B', (0, 255, 0, 0, 255, 0)))
    x_rot1, y_rot1 = rotate(15, 7.5, -env.car.angle)
    x_rot2, y_rot2 = rotate(15, -7.5, -env.car.angle)
    x_rot3, y_rot3 = rotate(-15, -7.5, -env.car.angle)
    x_rot4, y_rot4 = rotate(-15, 7.5, -env.car.angle)
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
    ('v2f', (env.car.x + x_rot1, env.car.y + y_rot1,
        env.car.x + x_rot2, env.car.y + y_rot2,
        env.car.x + x_rot3, env.car.y + y_rot3,
        env.car.x + x_rot4, env.car.y + y_rot4)),
    ('c3B', color)
    )
    x_rot1, y_rot1 = rotate(13, 6, -env.car.angle)
    x_rot2, y_rot2 = rotate(13, -6, -env.car.angle)
    x_rot3, y_rot3 = rotate(7.5, -6, -env.car.angle)
    x_rot4, y_rot4 = rotate(7.5, 6, -env.car.angle)
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
    ('v2f', (env.car.x + x_rot1, env.car.y + y_rot1,
        env.car.x + x_rot2, env.car.y + y_rot2,
        env.car.x + x_rot3, env.car.y + y_rot3,
        env.car.x + x_rot4, env.car.y + y_rot4)),
    ('c3B', (50, 50, 250, 50, 50, 250, 50, 50, 250, 50, 50, 250))
    )





train() #change to see in order to check progress
