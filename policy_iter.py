from env import states, actions, step, action_labels

def policy_evaluation(policy, gamma, theta=1e-6):
    V = {s:0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            if s in {(1,1),(1,2),(2,1),(2,3)}:
                continue
            a = policy[s]
            ns, r = step(s, a)
            v = r + gamma * V[ns]
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_iteration(gamma):
    policy = {}
    for s in states:
        if s in {(1,1),(1,2),(2,1),(2,3)}:
            policy[s] = None
        else:
            policy[s] = actions[0]
    stable = False
    iterations = 0
    while not stable:
        iterations += 1
        V = policy_evaluation(policy, gamma)
        stable = True
        for s in states:
            if s in {(1,1),(1,2),(2,1),(2,3)}:
                continue
            best_a = None
            best_v = None
            for a in actions:
                ns, r = step(s, a)
                val = r + gamma * V[ns]
                if best_v is None or val > best_v:
                    best_v = val
                    best_a = a
            if best_a != policy[s]:
                policy[s] = best_a
                stable = False
    return V, policy, iterations

if __name__ == "__main__":
    for g in [0.9,0.5,0.1]:
        V, pi, iters = policy_iteration(g)
        print("gamma",g,"iters",iters)
        for y in range(3,-1,-1):
            print(" ".join(f"{V[(x,y)]:6.2f}" for x in range(3)))
        print("policy")
        for y in range(3,-1,-1):
            print(" ".join(action_labels[pi[(x,y)]] if pi[(x,y)] else "·" for x in range(3)))
        print()

