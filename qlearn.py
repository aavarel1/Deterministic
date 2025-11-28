from env import states, actions, step, action_labels
import random

def q_learning(gamma, episodes=5000, alpha=0.1, eps=0.1):
    Q = {(s,a):0.0 for s in states for a in actions}
    for _ in range(episodes):
        s = random.choice([st for st in states if st not in {(1,1),(1,2),(2,1),(2,3)}])
        while s not in {(1,1),(1,2),(2,1),(2,3)}:
            if random.random() < eps:
                a = random.choice(actions)
            else:
                qs = [Q[(s,aa)] for aa in actions]
                m = max(qs)
                a = random.choice([aa for aa in actions if Q[(s,aa)] == m])
            ns, r = step(s, a)
            next_qs = [Q[(ns,aa)] for aa in actions]
            Q[(s,a)] += alpha * (r + gamma * max(next_qs) - Q[(s,a)])
            s = ns
    policy = {}
    for s in states:
        if s in {(1,1),(1,2),(2,1),(2,3)}:
            policy[s] = None
        else:
            policy[s] = max(actions, key=lambda a: Q[(s,a)])
    return Q, policy

if __name__ == "__main__":
    for g in [0.9,0.5,0.1]:
        Q, pi = q_learning(g)
        print("gamma",g)
        for y in range(3,-1,-1):
            row = []
            for x in range(3):
                if (x,y) in {(1,1),(1,2),(2,1),(2,3)}:
                    row.append("  0.00")
                else:
                    vals = [Q[((x,y),a)] for a in actions]
                    row.append(f"{max(vals):6.2f}")
            print(" ".join(row))
        print("policy")
        for y in range(3,-1,-1):
            print(" ".join(action_labels[pi[(x,y)]] if pi[(x,y)] else "·" for x in range(3)))
        print()
