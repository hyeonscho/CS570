# python scripts/generate_ref_max_score.py --env_name maze2d-xxlarge-v1
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diffuser
import d4rl
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import argparse
import diffuser.utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v0', help='Maze type. small or default')
    parser.add_argument('--num_episodes', type=int, default=100, help='Num samples to collect')
    args = parser.parse_args()

    render_config = utils.Config(
        "utils.Maze2dRenderer",
        savepath=("/root/diffuser_chain_hd/scripts/tmp/", "render_config.pkl"),
        env=args.env_name,
    )
    renderer = render_config()

    env = gym.make(args.env_name)
    env.seed(0)
    np.random.seed(0)
    controller = waypoint_controller.WaypointController(env.str_maze_spec)

    ravg = []
    dones_t = []
    for i in range(args.num_episodes):
        s = env.reset()
        returns = 0
        rollout = []
        latch = False
        for t in range(env._max_episode_steps):
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env.get_target())
            s, rew, _, _ = env.step(act)
            rollout.append(s.copy())
            if done and not latch: 
                dones_t.append(t)
                latch = True
            returns += rew
        ravg.append(returns)
        plan_rollout = [np.array(rollout)]
        renderer.composite(os.path.join('/root/diffuser_chain_hd/scripts/tmp', f'plan_rollout{i}_{dones_t[-1]}.png'), plan_rollout, ncol=1) if done else renderer.composite(os.path.join('/root/diffuser_chain_hd/scripts/tmp', f'plan_rollout{i}_NA.png'), plan_rollout, ncol=1)

    print(args.env_name, 'returns', np.mean(ravg))
    print(f'min steps to done: {np.min(dones_t)}')
    print(f'max steps to done: {np.max(dones_t)}')
    print(dones_t)

if __name__ == "__main__":
    main()