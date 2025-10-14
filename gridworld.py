# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
#
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In mc_control, sarsa, q_learning, and double q-learning once a terminal state is reached,
the environment should be (re)initialized by
    s = env.reset()
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)

A Logger instance is provided for each function, through which you can
visualize the process of the algorithm:
    logger.log(i, v, pi)     # update values and policy at step/iteration i
    logger.log(i, v)         # update only values
"""

import random


def value_iteration(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    theta = 1e-4  # convergence tolerance (infinity norm)

    def q_value(s, a, vref):
        total = 0.0
        for (p, s_next, r, terminal) in TRANSITION_MODEL[s][a]:
            cont = 0.0 if terminal else vref[s_next]
            total += p * (r + gamma * cont)
        return total

    for k in range(max_iterations):
        delta = 0.0
        new_v = v[:]
        for s in range(NUM_STATES):
            best_q = None
            best_a = 0
            for a in range(NUM_ACTIONS):
                q = q_value(s, a, v)
                if best_q is None or q > best_q or (q == best_q and a < best_a):
                    best_q, best_a = q, a
            new_v[s] = best_q
            pi[s] = best_a
            delta = max(delta, abs(new_v[s] - v[s]))
        v = new_v
        logger.log(k + 1, v, pi)
        if delta < theta:
            break
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0.0] * NUM_STATES
    # random per-state initial policy (not broadcasted)
    pi = [random.randint(0, NUM_ACTIONS - 1) for _ in range(NUM_STATES)]
    logger.log(0, v, pi)

    theta = 1e-4

    def q_value(s, a, vref):
        total = 0.0
        for p, s_next, r, terminal in TRANSITION_MODEL[s][a]:  # FIX: use TRANSITION_MODEL
            cont = 0.0 if terminal else vref[s_next]
            total += p * (r + gamma * cont)
        return total

    for k in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0.0
            new_v = v[:]
            for s in range(NUM_STATES):
                a = pi[s]
                new_v[s] = q_value(s, a, v)
                delta = max(delta, abs(new_v[s] - v[s]))
            v = new_v
            logger.log(k + 1, v, pi)
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(NUM_STATES):
            old_a = pi[s]
            best_q, best_a = None, old_a
            for a in range(NUM_ACTIONS):
                q = q_value(s, a, v)
                if best_q is None or q > best_q or (q == best_q and a < best_a):
                    best_q, best_a = q, a
            pi[s] = best_a
            if pi[s] != old_a:
                policy_stable = False

        logger.log(k + 1, v, pi)
        if policy_stable:
            break

    return pi


def on_policy_mc_control(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    # Q-table initialization (FIX)
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    alpha = 0.1
    min_eps = 0.05
    log_every_episode = 1

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            return best_a
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0
    episode_count = 0

    while steps_used < max_iterations:
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))
        s = env.reset()
        episode = []

        while True:
            a = epsilon_greedy_action(s, eps)
            s_next, r, terminal, _ = env.step(a)
            episode.append((s, a, r))
            steps_used += 1
            if terminal or steps_used >= max_iterations:
                break
            s = s_next

        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                Q[s_t][a_t] += alpha * (G - Q[s_t][a_t])

        for s in range(NUM_STATES):
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            pi[s] = best_a
            v[s] = best_q

        episode_count += 1
        if episode_count % log_every_episode == 0:
            logger.log(steps_used, v, pi)

    return pi


def sarsa(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    # Q-table initialization (FIX)
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    alpha = 0.1
    min_eps = 0.05

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            return best_a
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0
    while steps_used < max_iterations:
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))
        s = env.reset()
        a = epsilon_greedy_action(s, eps)

        while steps_used < max_iterations:
            s_next, r, terminal, _ = env.step(a)
            steps_used += 1

            if not terminal:
                a_next = epsilon_greedy_action(s_next, eps)
                target = r + gamma * Q[s_next][a_next]
            else:
                a_next = None
                target = r

            Q[s][a] += alpha * (target - Q[s][a])

            s, a = s_next, a_next
            if terminal:
                break

        for s_idx in range(NUM_STATES):
            best_a, best_q = 0, Q[s_idx][0]
            for a_idx in range(1, NUM_ACTIONS):
                if Q[s_idx][a_idx] > best_q or (Q[s_idx][a_idx] == best_q and a_idx < best_a):
                    best_a, best_q = a_idx, Q[s_idx][a_idx]
            pi[s_idx] = best_a
            v[s_idx] = best_q

        logger.log(steps_used, v, pi)

    return pi


def q_learning(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    # Q-table initialization (FIX)
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    alpha = 0.1
    min_eps = 0.05

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            return best_a
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0
    while steps_used < max_iterations:
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))
        s = env.reset()

        while steps_used < max_iterations:
            a = epsilon_greedy_action(s, eps)
            s_next, r, terminal, _ = env.step(a)
            steps_used += 1

            if terminal:
                target = r
            else:
                best_next = Q[s_next][0]
                for a2 in range(1, NUM_ACTIONS):
                    if Q[s_next][a2] > best_next:
                        best_next = Q[s_next][a2]
                target = r + gamma * best_next

            Q[s][a] += alpha * (target - Q[s][a])
            s = s_next
            if terminal:
                break

        for s_idx in range(NUM_STATES):
            best_a, best_q = 0, Q[s_idx][0]
            for a_idx in range(1, NUM_ACTIONS):
                if Q[s_idx][a_idx] > best_q or (Q[s_idx][a_idx] == best_q and a_idx < best_a):
                    best_a, best_q = a_idx, Q[s_idx][a_idx]
            pi[s_idx] = best_a
            v[s_idx] = best_q

        logger.log(steps_used, v, pi)

    return pi


def double_q_learning(env, gamma, max_iterations, logger):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    # Double Q initializations (FIX)
    Q1 = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    Q2 = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    alpha = 0.1
    min_eps = 0.05

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            best_a, best_q = 0, Q1[s][0] + Q2[s][0]
            for a in range(1, NUM_ACTIONS):
                q_sum = Q1[s][a] + Q2[s][a]
                if q_sum > best_q or (q_sum == best_q and a < best_a):
                    best_a, best_q = a, q_sum
            return best_a
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0
    while steps_used < max_iterations:
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))
        s = env.reset()

        while steps_used < max_iterations:
            a = epsilon_greedy_action(s, eps)
            s_next, r, terminal, _ = env.step(a)
            steps_used += 1

            if random.random() < 0.5:
                if terminal:
                    target = r
                else:
                    a_star = 0
                    best_q1 = Q1[s_next][0]
                    for a2 in range(1, NUM_ACTIONS):
                        if Q1[s_next][a2] > best_q1 or (Q1[s_next][a2] == best_q1 and a2 < a_star):
                            a_star, best_q1 = a2, Q1[s_next][a2]
                    target = r + gamma * Q2[s_next][a_star]
                Q1[s][a] += alpha * (target - Q1[s][a])
            else:
                if terminal:
                    target = r
                else:
                    a_star = 0
                    best_q2 = Q2[s_next][0]
                    for a2 in range(1, NUM_ACTIONS):
                        if Q2[s_next][a2] > best_q2 or (Q2[s_next][a2] == best_q2 and a2 < a_star):
                            a_star, best_q2 = a2, Q2[s_next][a2]
                    target = r + gamma * Q1[s_next][a_star]
                Q2[s][a] += alpha * (target - Q2[s][a])

            s = s_next
            if terminal:
                break

        for s_idx in range(NUM_STATES):
            best_a, best_q = 0, Q1[s_idx][0] + Q2[s_idx][0]
            for a_idx in range(1, NUM_ACTIONS):
                q_sum = Q1[s_idx][a_idx] + Q2[s_idx][a_idx]
                if q_sum > best_q or (q_sum == best_q and a_idx < best_a):
                    best_a, best_q = a_idx, q_sum
            pi[s_idx] = best_a
            v[s_idx] = best_q

        logger.log(steps_used, v, pi)

    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "On-policy MC Control": on_policy_mc_control,
        "SARSA": sarsa,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q_learning
    }

    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda: [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        "world3": lambda: [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o", 1, "_", 10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10],
        ],
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()
