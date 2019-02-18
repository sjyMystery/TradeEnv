from DDQN import DoubleDQN,DeepQNetwork
from tradenv.Env import MarketEnv
from tradenv.Feed import CSVFeed
import tensorflow as tf
import numpy as np
import datetime

tf.flags.DEFINE_integer('workers', 4, """co-workers num""", 1)
tf.flags.DEFINE_string('instrument', "USDJPY", """name of instruments""")
tf.flags.DEFINE_integer('per_bin', 1000, """log state per bin""", 1000)
tf.flags.DEFINE_integer('per_trade', 100, """log state per trade""", 100)
tf.flags.DEFINE_integer('time_length', 60 * 24 * 30, """time step length""", 1)
tf.flags.DEFINE_integer('eps', 10000, """""")
tf.flags.DEFINE_integer('update_steps', 1000, """""""")
tf.flags.DEFINE_string('start_date', '2012-01-01', """train begin date""")
tf.flags.DEFINE_string('end_date', '2013-01-01', """train end date""")

start_date = datetime.datetime.strptime(tf.flags.FLAGS.start_date, "%Y-%m-%d")
end_date = datetime.datetime.strptime(tf.flags.FLAGS.end_date, "%Y-%m-%d")
instrument = tf.flags.FLAGS.instrument
update_steps = tf.flags.FLAGS.update_steps
max_eps = tf.flags.FLAGS.eps
time_length = tf.flags.FLAGS.time_length

rate = 0.1


def action_convert(action_):
    return action_


MEMORY_SIZE = 1024

feed = CSVFeed()
feed.append_csv("./20.csv.gz")
env = MarketEnv(100000, feed, False)
dqn = DoubleDQN(3, env.observe_space.shape[0], memory_size=MEMORY_SIZE, replace_target_iter=300)
eps = 0
steps = 0
total = 0
while eps < max_eps:
    state = env.reset()
    done = False
    while not done:
        action = dqn.choose_action(np.array(state))
        action = action_convert(action)
        state_, rewards, done = env.step(action)

        dqn.store_transition(state, action, float(rewards) / 10000.0, state_)
        if total > MEMORY_SIZE:
            dqn.learn()
        state = state_
        steps += 1
        total += 1
        #
        # if steps % 1000 == 0:
        #     print(state)

    rate *= 0.9
    eps += 1
    print(env.equity, env.win_rate,env.win+env.loss)
    # dqn.plot_cost()
