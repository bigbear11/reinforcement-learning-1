import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000

#policy gradient的一种,REINFORCE算法
# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size#4
        self.action_size = action_size#2

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()#2
        return np.random.choice(self.action_size, 1, p=policy)[0]#choose action accordding to probability

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        '''
        example:
        self.states:[array([[-0.00647736, -0.04499117,  0.02213829, -0.00486359]]), array([[-0.00737719, -0.24042351,  0.02204101,  0.2947212 ]]), array([[-0.01218566, -0.04562261,  0.02793544,  0.00907036]]), array([[-0.01309811, -0.24113382,  0.02811684,  0.31043471]]), array([[-0.01792078, -0.04642351,  0.03432554,  0.02674995]]), array([[-0.01884925, -0.24202048,  0.03486054,  0.33006229]]), array([[-0.02368966, -0.04741166,  0.04146178,  0.04857336]]), array([[-0.0246379 , -0.24310286,  0.04243325,  0.35404415]]), array([[-0.02949995, -0.43880168,  0.04951413,  0.65979978]]), array([[-0.03827599, -0.2444025 ,  0.06271013,  0.38310959]]), array([[-0.04316404, -0.44035616,  0.07037232,  0.69488702]]), array([[-0.05197116, -0.63637999,  0.08427006,  1.00886738]]), array([[-0.06469876, -0.83251953,  0.10444741,  1.32677873]]), array([[-0.08134915, -0.63885961,  0.13098298,  1.06852366]]), array([[-0.09412634, -0.44569036,  0.15235346,  0.8196508 ]]), array([[-0.10304015, -0.25294509,  0.16874647,  0.57850069]]), array([[-0.10809905, -0.44997994,  0.18031649,  0.91923131]]), array([[-0.11709865, -0.25769299,  0.19870111,  0.68820344]])]
        self.rewards:[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -100]
        self.actions:[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
        '''
        episode_length = len(self.states)#18
        discounted_rewards = self.discount_rewards(self.rewards)
        '''
        example:
        disconnted_rewards:array([ -68.58863868,  -70.29155422,  -72.01167093,  -73.74916255,-75.5042046 , -77.27697434, -79.06765085, -80.876415  , -82.7034495 ,  -84.54893889,  -86.41306958,  -88.29602988,-90.19800998, -92.119202 , -94.0598,-96.02,-98., -100. ])
        '''
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)#将作为神经网络预测对象
        '''
        array([ 1.59468271,  1.41701722,  1.23755712,  1.05628429,  0.87318042,
        0.68822702,  0.50140541,  0.3126967 ,  0.12208185, -0.0704584 ,
       -0.26494351, -0.46139311, -0.65982705, -0.86026537, -1.06272832,
       -1.26723636, -1.47381013, -1.6824705 ])
        '''
        update_inputs = np.zeros((episode_length, self.state_size))#shape(18,4)
        advantages = np.zeros((episode_length, self.action_size))#shape(18,2)

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        import pdb; pdb.set_trace()
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_reinforce.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_reinforce.h5")
