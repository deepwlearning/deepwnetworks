# Deep W Networks

This is the repository for the paper entitled: "Deep W-Networks: Solving Multi-Objective Optimisation Problems With Deep Reinforcement Learning", currently under consideration for publication at The 15th International Conference on Agents and Artificial Intelligence (ICAART 2023). 

The proposed Deep W-Networks (DWN) algorithm takes advantage of the computational efficiency of single-policy algorithms by considering each objective separately. These policies will suggest a selfish action that will only maximise their reward. However, the DWN resolves the competition between greedy single-objective policies by relying on W-values representing policies’ value to the system. These W-values can be learned with interaction with the environment, following logical steps similar to a well-known Q-learning algorithm. In our proposed implementation, we employ two Deep Q-Networks (DQNs) for each objective. One DQN is used to learn a greedy policy for the given objective, while the second DQN has only one output representing the policy’s W-value for a given state input. Additionally, DWN has the benefit of training all policies simultaneously, which allows for a faster learning process.

## Running the code

The pretrained nets are for the DWN for the mountain car and deep sea environments. Running *DWN_mountain_car.py* and *DWN_deep_sea_treasure.py* shows the learned behaviour for the proposed DWM algorithm. To compare the behaviour to DQN, run *DQN_mountain_car.py* and *DWN_deep_sea_treasure.py*.


## Citing

Bibtex:
```
@inproceedings{hribar2023deep,
  title={Deep W-Networks: Solving Multi-Objective Optimisation Problems With Deep Reinforcement Learning},
  author={Hribar, Jernej and Hackett, Luke and Dusparic, Ivana},
  booktitle={ICAART 2023: 15th International Conference on Agents and Artificial Intelligence},
  year={2023}
}
```