# Policy Gradient Methods
這些算法統稱為策略梯度法。

研究策略梯度方法的一個基本動機是解決 Q-learning 的局限性。 我們會記得 Q-learning 是關於選擇最大化狀態值的動作。
使用 Q 函數，我們能夠確定使代理能夠決定針對給定狀態採取何種操作的策略。 所選擇的動作只是為代理提供最大值的動作。

在這方面，Q-learning 僅限於有限數量的離散動作。 它無法處理連續的動作空間環境。 此外，Q-learning 並不是直接優化策略。 
最後，強化學習是關於找到代理能夠使用的最佳策略，以便決定它應該採取什麼行動來最大化回報。

## Policy gradient theorem 
策略梯度定理

## Monte Carlo policy gradient
Monte Carlo policy gradient 
It does not require knowledge of the dynamics of the env (in other words,model-free)
only experience samples are needed to optimally tune the parameters of the policy network

## REINFORCE with baseline method 

## Actor -Critic method 
In the REINFORCE with baseline method, the value is used as a baseline. It is
not used to train the value function. In this section, we introduce a variation of REINFORCE with baseline, 
called the Actor-Critic method. The policy and value networks play the roles of actor and critic networks. 
The policy network is the actor deciding which action to take given the state. 
Meanwhile, the value network evaluates the decision made by the actor or policy network.
在 REINFORCE with baseline 方法中，該值用作基線。 這是
不用於訓練價值函數。 在本節中，我們介紹了 REINFORCE 與基線的變體，稱為 Actor-Critic 方法。
政策和價值網絡扮演演員和評論家網絡的角色。 政策網絡是決定在給定狀態下採取何種行動的參與者。 
同時，價值網絡評估參與者或政策網絡做出的決定。

our estimate is based on only on the immediate future,This is known as the bootstrappign technique

## Adavantage Actor-Critic(A2C) method 

## 