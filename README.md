# Frogger

This is a simple <b>DQN</b> implementation for frogger game. 

Game created using Unity and python API of Unity ML Agent [1] is used to run game with Keras implementation of DQN

Game for the windows system available in [2]

[1] - https://github.com/Unity-Technologies/ml-agents

[2] - https://drive.google.com/drive/folders/155Z71QdXSCa7XRmnzXggb1ZwJl-A6iZd?usp=sharing

Unity mlagent has a bug in linux platform that prevent worker from closing, thus when using in the linux e need to use 
separate worker id for observing and training 

```python
def reset(self):
    self.close()
    self.loadEnv(0) # change this to 1 in linux
```

issue : https://github.com/Unity-Technologies/ml-agents/issues/1505

workaround 
```python
env = UnityEnvironment(file_name=env_name, worker_id=0, seed=1)
```


