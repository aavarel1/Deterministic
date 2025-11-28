states = [(x, y) for y in range(4) for x in range(3)]
terminal_states = {(1,1), (1,2), (2,1), (2,3)}
rewards = {(1,1):-10, (2,1):-20, (1,2):10, (2,3):20}
actions = [(1,0), (-1,0), (0,1), (0,-1)]
action_labels = {(1,0):"→", (-1,0):"←", (0,1):"↑", (0,-1):"↓"}

def step(state, action):
    if state in terminal_states:
        return state, rewards[state]
    nx = state[0] + action[0]
    ny = state[1] + action[1]
    if 0 <= nx < 3 and 0 <= ny < 4:
        ns = (nx, ny)
    else:
        ns = state
    r = rewards.get(ns, 0)
    return ns, r

