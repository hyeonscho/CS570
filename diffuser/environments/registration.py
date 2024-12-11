import gym
from .xxlarge import XXLARGE_MAZE

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
        'id':'maze2d-xxlarge-v1',
        'entry_point':'d4rl.pointmaze:MazeEnv',
        'max_episode_steps':1300,
        'kwargs':{
            'maze_spec':XXLARGE_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 3.28,
            'ref_max_score': 310.1,
            # 'dataset_url': 'file:///data/datasets/d4rl/maze2d-xxlarge-sparse-v1.hdf5'
            'dataset_url': 'file:///root/diffuser_chain_hd/data/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url': 'file:///home/hany/repos/research/docker_wrapper/diffuser_chain_hd/maze2d-xxlarge-v1-sparse.hdf5'
            # 'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
        }
    }

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
