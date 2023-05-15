# RL 
Reinforcement Learning (RL) is a framework that is used by an agent for decision making. T
強化學習 (RL) 是代理用於決策制定的框架。


RL has been around for decades. However, beyond simple world models, RL has struggled to scale. This is where Deep Learning (DL) came into play.
RL 已經存在了幾十年。 然而，除了簡單的世界模型之外，強化學習還難以擴展。 這就是深度學習 (DL) 發揮作用的地方。
It solved this scalability problem, which opened up the era of Deep Reinforcement Learning (DRL).
它解決了這個可擴展性問題，從而開啟了深度強化學習（DRL）時代。

## Princple of RL 
The goal of RL is to learn the optimal policy that helps the robot to decide which action to take given a state to maximize the accumulated discounted reward:
RL 的目標是學習最優策略，幫助機器人決定在給定狀態下採取哪個動作以最大化累積的折扣獎勵：
the RL problem can be described as a Markov decision process (MDP).
RL 問題可以描述為馬爾可夫決策過程 (MDP)。

~~For simplicity, we'll assume a deterministic environment where a certain action in a given state will consistently result in a known next state and reward. In a later section of this chapter, we'll look at how to consider stochasticity. At timestep t:~~
~~為簡單起見，我們將假設一個確定性環境，在該環境中，給定狀態下的某個動作將始終導致已知的下一個狀態和獎勵。 在本章的後面部分，我們將研究如何考慮隨機性。 在時間步 t：~~
~~- The env is in a state.~~
## The Q value 
instead of finding the policy that maximaized the values for all states,
but to maximizes the quality(Q) value for all states 

If, for every action, the reward and the next state can be observed, we can formulate the following iterative or trial-and-error algorithm to learn the Q value:
$$
Q(s,a) = r + ymax(s^,a^,)
$$
`s'`和`a'`是下一个状态和动作。

上面公式is known as the bellman equation,which is the core of the Q-learning algorithm.Q-learning attempts to approximate the first-order expansion of retunr of value  as a function of both current state and action.

被稱為貝爾曼方程，它是 Q-learning 算法的核心。Q-learning 試圖將價值回報率的一階展開近似為當前狀態和動作的函數。





## Q-learning example 
## Nondeterministic enviroment
## Temporal-difference learning 
## Deep Q-Network
## Conclusion
