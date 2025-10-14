# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)


"""
In this file, you should test your SARSA and Q-learning implementation on a robot crawler environment. 
It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

def sarsa(env, logger):
    """
    Implement SARSA to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    alpha = 0.1
    min_eps = 0.05
    max_iterations = 5000  # total steps budget

    # Control tables and logging buffers
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES

    # Initial visualization
    logger.log(0, v, pi)

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            # greedy with stable tie-breaking
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            return best_a
        return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0

    while steps_used < max_iterations:
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))

        s = env.reset()
        a = epsilon_greedy_action(s, eps)

        while steps_used < max_iterations:
            s_next, r, terminal, _ = env.step(a)
            steps_used += 1

            if terminal:
                target = r
                Q[s][a] += alpha * (target - Q[s][a])
                break
            else:
                a_next = epsilon_greedy_action(s_next, eps)
                target = r + gamma * Q[s_next][a_next]
                Q[s][a] += alpha * (target - Q[s][a])
                s, a = s_next, a_next

        for s_idx in range(NUM_STATES):
            best_a, best_q = 0, Q[s_idx][0]
            for a_idx in range(1, NUM_ACTIONS):
                if Q[s_idx][a_idx] > best_q or (Q[s_idx][a_idx] == best_q and a_idx < best_a):
                    best_a, best_q = a_idx, Q[s_idx][a_idx]
            pi[s_idx] = best_a
            v[s_idx] = best_q

        logger.log(steps_used, v, pi)


    return pi

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    alpha = 0.1
    min_eps = 0.05
    max_iterations = 5000  # total steps budget

    # Control tables and logging buffers
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES

    # Initial visualization
    logger.log(0, v, pi)

    def epsilon_greedy_action(s, eps):
        if random.random() > eps:
            best_a, best_q = 0, Q[s][0]
            for a in range(1, NUM_ACTIONS):
                if Q[s][a] > best_q or (Q[s][a] == best_q and a < best_a):
                    best_a, best_q = a, Q[s][a]
            return best_a
        return random.randint(0, NUM_ACTIONS - 1)

    steps_used = 0

    while steps_used < max_iterations:
        # GLIE-style epsilon decay by steps
        eps = max(min_eps, 1.0 - (steps_used / float(max_iterations)))

        s = env.reset()

        while steps_used < max_iterations:
            a = epsilon_greedy_action(s, eps)
            s_next, r, terminal, _ = env.step(a)
            steps_used += 1

            if terminal:
                target = r
            else:
                # max_a' Q(s', a')
                best_next = Q[s_next][0]
                for a2 in range(1, NUM_ACTIONS):
                    if Q[s_next][a2] > best_next:
                        best_next = Q[s_next][a2]
                target = r + gamma * best_next

            Q[s][a] += alpha * (target - Q[s][a])
            s = s_next

            if terminal:
                break

        # Improve deterministic policy and compute V for logging
        for s_idx in range(NUM_STATES):
            best_a, best_q = 0, Q[s_idx][0]
            for a_idx in range(1, NUM_ACTIONS):
                if Q[s_idx][a_idx] > best_q or (Q[s_idx][a_idx] == best_q and a_idx < best_a):
                    best_a, best_q = a_idx, Q[s_idx][a_idx]
            pi[s_idx] = best_a
            v[s_idx] = best_q

        logger.log(steps_used, v, pi)

    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
         "SARSA": sarsa
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()