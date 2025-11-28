from env import states, actions, step, action_labels

def value_iteration(gamma, theta=1e-6):
    V = {s:0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            if s in {(1,1),(1,2),(2,1),(2,3)}:
                continue
            best = None
            for a in actions:
                ns, r = step(s, a)
                val = r + gamma * V[ns]
                if best is None or val > best:
                    best = val
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    policy = {}
    for s in states:
        if s in {(1,1),(1,2),(2,1),(2,3)}:
            policy[s] = None
            continue
        best_a = None
        best_v = None
        for a in actions:
            ns, r = step(s, a)
            val = r + gamma * V[ns]
            if best_v is None or val > best_v:
                best_v = val
                best_a = a
        policy[s] = best_a
    return V, policy

if __name__ == "__main__":
    for g in [0.9,0.5,0.1]:
        V, pi = value_iteration(g)
        print("gamma",g)
        for y in range(3,-1,-1):
            print(" ".join(f"{V[(x,y)]:6.2f}" for x in range(3)))
        print("policy")
        for y in range(3,-1,-1):
            print(" ".join(action_labels[pi[(x,y)]] if pi[(x,y)] else "·" for x in range(3)))
        print()
