# Self-Driving-Car
This is a repository containing my Self Driving Car project. It is based on a self driving car game made by an anonymous discord user called EMJzero who had problems with implementing AI to it. Link to his game that you can play: https://pastebin.com/LaaC9p2x

Required libraries:
- tensorflow (version 1.12.0, recommended to use tensorflow-gpu and theano)
- keras (version 2.2.2)
- numpy (version 1.15.1)
- scipy (version 1.1.0)
- noise (version 1.2.2)
- pyglet (version 1.3.2)
- vectormath (version 0.2.1)

Description of training:
  Training starts with epsilon of 1.0 which means all actions are random. It is then being lowered after every epoch until it reaches a minEpsilon value which is set by user. Model is being trained every iteration on a batch of memory records. Both batch size and memory size can be set by user. Then neural network is being saved to some filepath every n epochs. The rewards are: -0.03 as a default reward, -1 for hitting wall and 2 for crossing checkpoints which are some lines along the track. All rewards can be changed deep in code.

The model that is saved in this repository called 'model.h5' learnt its way of playstyle after around 20-30 min, still having epsilon of around 0.4 meaning that 40% of his actions were random.
