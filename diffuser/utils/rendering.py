import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb
from math import pi
import torch

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment
from d4rl.pointmaze import maze_model

# -----------------------------------------------------------------------------#
# ------------------------------- helper structs ------------------------------#
# -----------------------------------------------------------------------------#

def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    if "giant" in env_id:
        maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]

def env_map(env_name):
    """
    map D4RL dataset names to custom fully-observed
    variants for rendering
    """
    if "halfcheetah" in env_name:
        return "HalfCheetahFullObs-v2"
    elif "hopper" in env_name:
        return "HopperFullObs-v2"
    elif "walker2d" in env_name:
        return "Walker2dFullObs-v2"
    elif 'stitched' in env_name:
        return env_name.split('stitched-')[1]
    else:
        return env_name


# -----------------------------------------------------------------------------#
# ------------------------------ helper functions -----------------------------#
# -----------------------------------------------------------------------------#


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)


def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype="uint8").reshape((height, width, 4))


# -----------------------------------------------------------------------------#
# ---------------------------------- renderers --------------------------------#
# -----------------------------------------------------------------------------#


class MuJoCoRenderer:
    """
    default mujoco renderer
    """

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print(
                "[ utils/rendering ] Warning: could not initialize offscreen renderer"
            )
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate(
            [
                np.zeros(1),
                observation,
            ]
        )
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate(
            [
                xpos[:, None],
                observations,
            ],
            axis=-1,
        )
        return states

    def render(
        self,
        observation,
        dim=256,
        partial=False,
        qvel=True,
        render_kwargs=None,
        conditions=None,
    ):
        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                "trackbodyid": 2,
                "distance": 3,
                "lookat": [xpos, -0.5, 1],
                "elevation": -20,
            }

        for key, val in render_kwargs.items():
            if key == "lookat":
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        render_kwargs = {
            "trackbodyid": 2,
            "distance": 10,
            "lookat": [5, 2, 0.5],
            "elevation": 0,
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(
                to_np(path),
                dim=dim,
                partial=True,
                qvel=True,
                render_kwargs=render_kwargs,
                **kwargs,
            )
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f"Saved {len(paths)} samples to: {savepath}")

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:, :-1]

        images_pred = np.stack(
            [self._renders(obs_pred, partial=True) for obs_pred in observations_pred]
        )

        images_real = np.stack(
            [self._renders(obs_real, partial=False) for obs_real in observations_real]
        )

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        """
        diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        """
        render_kwargs = {
            "trackbodyid": 2,
            "distance": 10,
            "lookat": [10, 2, 0.5],
            "elevation": 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f"[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}")

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[
                :, :, : self.observation_dim
            ]

            frame = []
            for states in states_l:
                img = self.composite(
                    None,
                    states,
                    dim=(1024, 256),
                    partial=True,
                    qvel=True,
                    render_kwargs=render_kwargs,
                )
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


# -----------------------------------------------------------------------------#
# ----------------------------------- maze2d ----------------------------------#
# -----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
    "maze2d-clarge-v1": (0, 9, 0, 12),
    "maze2d-ultra-v1": (0, 12, 0, 16),
    "maze2d-giant-v1": (0, 12, 0, 16),
    "maze2d-xxlarge-v1": (0, 18, 0, 24),
    "maze2d-xxlarge-v2": (0, 18, 0, 24),
    "maze2d-xxlarge-v3": (0, 18, 0, 24),
    "maze2d-xxlargec-v1": (0, 18, 0, 24),

    'pointmaze-medium-navigate-v0': (0, 8, 0, 8),
    'pointmaze-large-navigate-v0': (0, 9, 0, 12),
    'pointmaze-giant-navigate-v0': (0, 12, 0, 16),
    'pointmaze-teleport-navigate-v0': (0, 9, 0, 12), 
    'pointmaze-medium-stitch-v0': (0, 8, 0, 8), 
    'pointmaze-large-stitch-v0': (0, 9, 0, 12),
    'pointmaze-giant-stitch-v0': (0, 12, 0, 16),
    'pointmaze-teleport-stitch-v0': (0, 9, 0, 12),

    'antmaze-medium-navigate-v0': (0, 8, 0, 8),
    'antmaze-large-navigate-v0': (0, 9, 0, 12),
    'antmaze-giant-navigate-v0': (0, 12, 0, 16),
    'antmaze-teleport-navigate-v0': (0, 9, 0, 12), 
    'antmaze-medium-stitch-v0': (0, 8, 0, 8), 
    'antmaze-large-stitch-v0': (0, 9, 0, 12),
    'antmaze-giant-stitch-v0': (0, 12, 0, 16),
    'antmaze-teleport-stitch-v0': (0, 9, 0, 12),
}


class MazeRenderer:
    def __init__(self, env):
        if type(env) is str:
            env = load_environment(env)
            self._config = env._config
            self._background = self._config != " "
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        map_height, map_width = self._background.shape[:2]
        fig.set_size_inches(map_width, map_height)  # 크기 조정 비율은 필요에 따라 변경
        # fig.set_size_inches(5, 5)

        # def convert_maze_string_to_grid(maze_string):
        #     lines = maze_string.split("\\")
        #     grid = [line[1:-1] for line in lines]
        #     return grid[1:-1]
        # if "giant" in cfg.dataset:
        #    maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
        # maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
        # grid = convert_maze_string_to_grid(maze_string)
        # plt.figure()
        plt.scatter(observations[:, 0]/4+1, observations[:, 1]/4+1, c=np.arange(len(observations[:])), cmap="Reds")

        # plt.scatter(observations[:, 0]/4+1, observations[:, 1]/4+1, c=np.arange(len(observations[:])), cmap="Reds")

        #plt.scatter(observations[:, 0]/4+1, observations[:, 1]/4+1, c=np.arange(len(observations[:])), cmap="Reds")
        # for i, row in enumerate(grid):
        #     for j, cell in enumerate(row):
        #         if cell == "#":
        #             square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
        #             plt.gca().add_patch(square)
        # plt.gca().set_aspect("equal", adjustable="box")
        # plt.gca().set_facecolor("lightgray")
        # plt.gca().set_axisbelow(True)
        # plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
        # plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
        # plt.xlim([0.5, len(grid) + 0.5])
        # plt.ylim([0.5, len(grid[0]) + 0.5])
        # plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        # plt.grid(True, color="white", which="minor", linewidth=4)
        # plt.gca().spines["top"].set_linewidth(4)
        # plt.gca().spines["right"].set_linewidth(4)
        # plt.gca().spines["bottom"].set_linewidth(4)
        # plt.gca().spines["left"].set_linewidth(4)

        # if self._background is not None:
        #     height, width = self._background.shape[:2]
        #     self._extent = (0, width, height, 0)
            
        plt.imshow(self._background * 0.5,cmap=plt.cm.binary,vmin=0,vmax=1,)

        # path_length = len(observations)
        # colors = plt.cm.jet(np.linspace(0, 1, path_length))
        # plt.plot(observations[:, 1], observations[:, 0], c="black", zorder=10)
        # plt.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)
        plt.axis("off")
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        savepath : str
        observations : [ n_paths x horizon x 2 ]
        """
        assert (
            len(paths) % ncol == 0
        ), "Number of paths must be divisible by number of columns"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)

        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncol
        )
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")


class Maze2dRenderer(MazeRenderer):
    def __init__(self, env, observation_dim=None):
        if 'stitched' in env:
            env = env.split('stitched-')[1]
        self.env_name = env
        self.env = load_environment(env)
        # self._background = self.env.maze_arr == 10
        self._background = self.env.maze_map == 1
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        # observations = observations + 1
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            self._extent = (-0.5, iscale - 0.5, jscale - 0.5, -0.5)
            # observations[:, 0] = (observations[:, 0] + 4) / iscale
            # observations[:, 1] = (observations[:, 1] + 4) / jscale
        else:
            raise RuntimeError(f"Unrecognized bounds for {self.env_name}: {bounds}")

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)


# -----------------------------------------------------------------------------#
# ---------------------------------- rollouts ---------------------------------#
# -----------------------------------------------------------------------------#


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f"[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, "
            f"but got state of size {state.size}"
        )
        state = state[: qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack(
        [rollout_from_state(env, state, actions) for actions in actions_l]
    )
    return rollouts


def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions) + 1):
        ## if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)

class CubeRenderer:
    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = observation_dim
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._remove_margins = False
        # self._extent = (0, 1, 1, 0)

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        savepath : str
        observations : [ n_paths x horizon x 2 ]
        """
        assert (
            len(paths) % ncol == 0
        ), "Number of paths must be divisible by number of columns"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)

        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncol
        )
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")

    def renders(self, observations, conditions=None, **kwargs):
        num_poses = observations.shape[-1] // 3
        # 3D plot
        fig = plt.gcf()

        # background as gray
        ax = plt.axes(projection='3d')
        ax.set_facecolor("gray")
        # set angle
        ax.view_init(elev=20, azim=30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        cmaps = ["Reds", "Blues", "Greens", "Oranges", "Purples", "Greys", "YlOrBr", "YlGn", "YlGnBu", "YlOrRd"]
        for i in range(num_poses):
            ax.scatter(observations[:, i*3], observations[:, i*3+1], observations[:, i*3+2], c=np.arange(len(observations)), cmap=cmaps[i])
        img = plot2img(fig, remove_margins=self._remove_margins)

        return img
    
class VisRenderer:
    def __init__(self, env, observation_dim=None, dataset=None):
        from ogbench.pretrain.models.mlp import MLP
        from ogbench.pretrain.models.bvae import BetaVAE

        self.env_name = env
        pretrained_model_path = '/home/baek1127/ogbench/embedded_data/'
        if 'medium' in self.env_name:
            pretrained_model_path += 'medium'
        elif 'large' in self.env_name:
            pretrained_model_path += 'large'
        elif 'giant' in self.env_name:
            pretrained_model_path += 'giant'
        self.invd_model_path = pretrained_model_path + '/invd.pth'
        self.e2s_model_path = pretrained_model_path + '/e2s.pth'
        self.vae_model_path = pretrained_model_path + '/vae.pth'
        # self.env = load_environment(env)
        self.observation_dim = observation_dim
        self.action_dim = 0
        self.dataset = dataset

        # self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._remove_margins = False
        # self._extent = (0, 1, 1, 0)
        


        self.action_model = MLP(self.observation_dim * 3, 2, hidden_dim=1024, num_layers=3) # use 3 frames
        self.action_model.load_state_dict(torch.load(self.invd_model_path))

        self.e2s_model = MLP(self.observation_dim, 2, hidden_dim=1024, num_layers=4) # position 2
        self.e2s_model.load_state_dict(torch.load(self.e2s_model_path))

        self.vae_model = BetaVAE()
        self.vae_model.load_state_dict(torch.load(self.vae_model_path))

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        savepath : str
        observations : [ n_paths x horizon x 2 ]
        """
        assert (
            len(paths) % ncol == 0
        ), "Number of paths must be divisible by number of columns"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)

        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncol
        )
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")
        pass

    def renders(self, observations, conditions=None, **kwargs):

        # observations = self.dataset.normalizer.normalize(torch.tensor(observations), "observations")

        observations = self.decode_position_from_normalized_emb(observations)

        plot_end_points = kwargs.get('plot_end_points', False)
        start = kwargs.get('start', None)
        goal = kwargs.get('goal', None)
        
        plt.clf()
        fig = plt.gcf()

        ax = plt.axes()
        
        maze_grid = get_maze_grid(self.env_name)
        plot_maze_layout(ax, maze_grid)
        ax.scatter(observations[:, 0], observations[:, 1], c=np.arange(len(observations)), cmap="Reds")
        
        if plot_end_points:
            start_goal = (start, goal)
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)

        plt.close()

        return img
    
    def decode_position_from_normalized_emb(self, emb):
        # Normalizations for medium maze
        emb = torch.tensor(emb).to(torch.float32)
        pos = self.e2s_model(emb).detach().cpu().numpy()

        if 'medium' in self.env_name and 'point' in self.env_name:
            pos_mean = np.array([10.273524, 9.648321])
            pos_std = np.array([5.627576, 4.897987])
        elif 'large' in self.env_name and 'point' in self.env_name:
            pos_mean = np.array([16.702621, 10.974173])
            pos_std = np.array([10.050303, 6.8203936])
        elif 'giant' in self.env_name and 'point' in self.env_name:
            pos_mean = np.array([24.888689, 17.158426])
            pos_std = np.array([14.732276, 11.651127])

        pos = pos * pos_std + pos_mean
        pos = pos / 4 + 1
        if not 'giant' in self.env_name:
            if len(pos.shape) == 2:
                pos = pos[:, [1, 0]]
            else: # if batched
                pos = pos[:, :, [1, 0]]
        return pos

def plot_start_goal(ax, start_goal: None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0
        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )
        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)
    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)
    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)

def plot_maze_layout(ax, maze_grid):
    ax.clear()
    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)
    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)