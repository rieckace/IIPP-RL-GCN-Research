# Experiment Register

This file is the authoritative index for new experiments.

| ID | Question | Model(s) | Training split | Evaluation split | Seeds | Status |
|---|---|---|---|---|---|---|
| E0 | Reproduce current GCN-DQN | GCN-Double-DQN | historical/current declared config | 5 maps | TBD | Pending |
| E1 | CNN vs graph representation | CNN-DQN, GCN-DQN | matched | matched | TBD | Pending |
| E2 | Topology generalization | CNN-DQN, GCN-DQN | declared subset | held-out maps | TBD | Pending |
| E3 | Scale shift | CNN-DQN, GCN-DQN | smaller layouts | larger layouts | TBD | Pending |
| E4 | Hazard shift | CNN-DQN, GCN-DQN | hazard regime A | hazard regimes B/C | TBD | Pending |
| E5 | Reward contribution | GCN-DQN variants | matched | matched | TBD | Pending |
| E6 | Graph representation | GCN variants | matched | matched | TBD | Pending |
| E7 | GCN depth | 2/3/5 layer variants | matched | matched | TBD | Pending |
| E8 | Improved architecture | selected candidate | matched | matched | TBD | Pending |

## Rule for every completed experiment

Record:
- exact git commit;
- config file;
- model checkpoint;
- random seeds;
- number of episodes;
- map split;
- hazard parameters;
- evaluation protocol;
- raw result file;
- generated figures;
- interpretation;
- limitations.
