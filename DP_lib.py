import copy
import numpy as np

# Random policy generation
def generate_random_policy(S_n, A_n):
    return np.ones([S_n, A_n]) / A_n

def policy_evaluation(env, policy, gamma=0.95, theta=1e-8):
    r"""Policy evaluation function. Loop until state values stable, delta < theta.

    Returns V comprising values of states under given policy.

    Args:
        env (gym.env): OpenAI environment class instantiated and assigned to an object.
        policy (np.array): policy array to evaluate
        gamma (float): discount rate for rewards
        theta (float): tiny positive number, anything below it indicates value function convergence
    """
    # 1. Create state-value array (16,)
    S_n = 16
    V = np.zeros(S_n)
    while True:
        delta = 0

        # 2. Loop through states
        for s in range(S_n):
            Vs = 0

            # 2.1 Loop through actions for the unique state
            # Given each state, we've 4 actions associated with different probabilities
            # 0.25 x 4 in this case, so we'll be looping 4 times (4 action probabilities) at each state
            for a, action_prob in enumerate(policy[s]):
                # 2.1.1 Loop through to get transition probabilities, next state, rewards and whether the game ended
                for prob, next_state, reward, done in env.P[s][a]:
                    # State-value function to get our values of states given policy
                    Vs += action_prob * prob * (reward + gamma * V[next_state])

            # This simple equation allows us to stop this loop when we've converged
            # How do we know? The new value of the state is smaller than a tiny positive value we set
            # State value change is tiny compared to what we have so we just stop!
            delta = max(delta, np.abs(V[s]-Vs))

            # 2.2 Update our state value for that state
            V[s] = Vs

        # 3. Stop policy evaluation if our state values changes are smaller than our tiny positive number
        if delta < theta:
            break

    return V

def q_value(env, V, s, gamma=0.95):
    r"""Q-value (action-value) function from state-value function

    Returns Q values, values of actions.

    Args:
        env (gym.env): OpenAI environment class instantiated and assigned to an object.
        V (np.array): array of state-values obtained from policy evaluation function.
        s (integer): integer representing current state in the gridworld
        gamma (float): discount rate for rewards.
    """
    # 1. Create q-value array for one state
    # We have 4 actions, so let's create an array with the size of 4
    A_n = 4
    q = np.zeros(A_n)

    # 2. Loop through each action
    for a in range(A_n):
        # 2.1 For each action, we've our transition probabilities, next state, rewards and whether the game ended
        for prob, next_state, reward, done in env.P[s][a]:
            # 2.1.1 Get our action-values from state-values
            q[a] += prob * (reward + gamma * V[next_state])

    # Return action values
    return q

def policy_improvement(env, V, gamma=0.95):
    r"""Function to improve the policy by utilizing state values and action (q) values.

    Args:
        env (gym.env): OpenAI environment class instantiated and assigned to an objects
        V (np.array): array of state-values obtained from policy evaluation function
        gamma (float): discount of rewards
    """
    # 1. Blank policy
    policy = np.zeros([env.nS, env.nA]) / env.nA

    # 2. For each state in 16 states
    for s in range(env.nS):

        # 2.1 Get q values: q.shape returns (4,)
        q = q_value(env, V, s, gamma)

        # 2.2 Find best action based on max q-value
        # np.argwhere(q==np.max(q)) gives the position of largest q value
        # given array([0.00852356, 0.01163091, 0.0108613 , 0.01550788]), this would return array([[3]]) of shape (1, 1)
        # .flatten() reduces the shape to (1,) where we've array([3])
        best_a = np.argwhere(q==np.max(q)).flatten()

        # 2.3 One-hot encode best action and store into policy array's row for that state
        # In our case where the best action is array([3]), this would return
        # array([0., 0., 0., 1.]) where position 3 is the best action
        # Now we can store the best action into our policy
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)

    return policy


def policy_iteration(env, gamma=0.95, theta=1e-8):
    # 1. Create equiprobable policy where every state has 4 actions with equal probabilities as a starting policy
    policy = np.zeros([env.nS, env.nA]) / env.nA
    # policy = generate_random_policy(env.nS, env.nA)
    state_value = []
    state_value.append(0)
    # 2. Loop through policy_evaluation and policy_improvement functions
    while True:
        # 2.1 Get state-values
        V = policy_evaluation(env, policy, gamma, theta)
        state_value.append(np.max(V[2]))
        # 2.2 Get new policy by getting q-values and maximizing q-values per state to get best action per state
        new_policy = policy_improvement(env, V, gamma)
        # state_value.append(np.max(new_policy[2]))
        # 2.3 Stop if the value function estimates for successive policies has converged
        if np.max(abs(policy_evaluation(env, policy, gamma) - policy_evaluation(env, new_policy, gamma))) < theta * 1e2:
            break
        # 2.4 Replace policy with new policy
        policy = copy.copy(new_policy)
    return policy, V, state_value
