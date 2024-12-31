import gym
from d4rl.pointmaze.maze_model import LARGE_MAZE
from .xxlarge import XXLARGE_MAZE
from .ultra import ULTRA_MAZE
from .giant import GIANT_MAZE

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
    },
    {
        'id':'maze2d-clarge-v1',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':800,
        'kwargs':{
            'maze_spec':LARGE_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 6.7,
            'ref_max_score': 273.99,
            'dataset_url': 'file:///root/diffuser_chain_hd/data/maze2d-large-v1-sparse.hdf5'
        }
    },
    
    {
        # can be solved under 600 steps
        'id':'maze2d-xxlarge-v1',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1300,
        'kwargs':{
            'maze_spec':XXLARGE_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 3.28,
            'ref_max_score': 310.1,
            'dataset_url': 'file:///root/diffuser_chain_hd/data/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    },
    
    # 8M version
    {
        # can be solved under 600 steps
        'id':'maze2d-xxlarge-v2',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1300,
        'kwargs':{
            'maze_spec':XXLARGE_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 3.28,
            'ref_max_score': 310.1,
            'dataset_url': 'file:///root/diffuser_chain_hd/data/maze2d-xxlarge-8M-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    },
    # Custom target points
    {
        # can be solved under 600 steps
        'id':'maze2d-xxlarge-v3',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1300,
        'kwargs':{
            'maze_spec':XXLARGE_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 3.28,
            'ref_max_score': 310.1,
            'dataset_url': 'file:///root/diffuser_chain_hd/data/maze2d-xxlarge-<>-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    },
    
    
    {
        'id':'maze2d-ultra-v1',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1000,
        'kwargs':{
            'maze_spec':ULTRA_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 15.04,
            'ref_max_score': 549.99,
            # 'dataset_url': 'file:///data/datasets/d4rl/maze2d-xxlarge-sparse-v1.hdf5'
            'dataset_url': 'file:///root/diffuser_chain_hd/maze2d-<>-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    },
    
    {
        'id':'maze2d-giant-v1',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1000,
        'kwargs':{
            'maze_spec':GIANT_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 9.63,
            'ref_max_score': 202.35,
            # 'dataset_url': 'file:///data/datasets/d4rl/maze2d-xxlarge-sparse-v1.hdf5'
            'dataset_url': 'file:///root/diffuser_chain_hd/maze2d-<>-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    },



)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()
