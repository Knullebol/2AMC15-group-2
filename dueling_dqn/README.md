----------------------------------------------------------
Vanilla DQN problems that we aim to solve using Double DQN and Dueling DQN 
----------------------------------------------------------
1. Reducing overestimation bias with Double DQN:

* Problem in vanilla DQN

    When computing the TD-target, we use the same network (or its clone) both to pick the best next action AND to evaluate its value. Due to estimation noise, that "max" tends to overshoot systematically (i.e., we over-estimate Q-values).

* Key idea of Double DQN

    We decouple action selection from action evaluation:
    - Selection by the online network
    - Evaluation by the target network

    By splitting those jobs, we should dramattically reduce 'optimistic' bias in our targets and hence learn more stable value estimates.

2. Decoupling State-Value from Action-Advantage with Dueling DQN:

* Problem in vanilla DQN

    In some states, choice of action hardly matters (e.g. when far away from any rewards), yet a standard Q-network still has to learn a seperate output for every action, making it inefficient at representing 'how good' the state is vs. 'how much better' one action is than another.

* Key idea of Dueling DQN

    We split the network into two streams after a shared feature extractor:
    - Value stream V(s), which estimates the value of being in state s.
    - Advantage stream A(s,a), which estimates how much better action a is compared to the average in that state.

    We then recombine into Q-values via for example, the 'average' aggregation. This lets the model state-value and relative action preferences in parallel, which speeds up learning, especially when many actions have similar value.
