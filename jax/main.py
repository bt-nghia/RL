import argparse
import gym
import numpy as np
from ddpg import DDPG
from td3 import TD3
from replay_buffer import ReplayBuffer


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed+100)

    avg_rw = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state)
            state, rw, done, _ = eval_env.step(action)
            avg_rw += rw

    avg_rw /= eval_episodes
    print("----------------------------")
    print("EVALUATION AVERAGE REWARD: %.2f" % (avg_rw))
    print("----------------------------")
    return avg_rw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v1")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--gamma", default=0.99, type=float)        # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    print("-----------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}")
    print("-----------------------------")

    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "input_dim": input_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": args.gamma,
        "tau": args.tau,
    }
    if args.policy == "DDPG":
        policy = DDPG(**kwargs)
    elif args.policy == "TD3":
        policy = TD3(**kwargs)
    else:
        raise ValueError("INVALID POLICY")
    
    replay_buffer = ReplayBuffer(input_dim, action_dim)
    evaluations = [eval_policy(policy, args.env, args.seed)]


    state, done = env.reset(), False
    episode_reward = 0.
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps+=1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state) + \
                np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, +max_action)

        next_state, rw, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0.

        replay_buffer.add(state, action, next_state, rw, done_bool)
        state = next_state
        episode_reward += rw

        if t > args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(f"TOTAL T: {t+1} EPISODE NUM: {episode_num+1} EPISODE T: {episode_timesteps} REWARD: {episode_reward:.2f}")
            state, done = env.reset(), False
            episode_reward = 0.
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))