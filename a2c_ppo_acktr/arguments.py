import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--carla-ego-vehicle-filter', default='vehicle.lincoln*', help='blueprint for ego vehicle')
    parser.add_argument(
        '--carla-town', default='Town02', help='which town to simulate: Town01 | Town02 | Town03')
    parser.add_argument(
        '--carla-task', default='random', help='mode of the task: random | roundabout')
    parser.add_argument(
        '--carla-discrete-action',
        action='store_true',
        default=False,
        help='use discrete action in carla')
    parser.add_argument(
        "--carla-port",
        action="store",
        type=int,
        default=2000,
        help="carla connection port")
    parser.add_argument(
        "--carla-display-size",
        action="store",
        type=int,
        default=256,
        help="screen size of bird-eye render")
    parser.add_argument(
        "--carla-number-of-vehicles",
        action="store",
        type=int,
        default=150,
        help="the number of vehicles in carla")
    parser.add_argument(
        "--carla-number-of-walkers",
        action="store",
        type=int,
        default=0,
        help="the number of walkers in carla")

    parser.add_argument(
        "--carla-max-past-step",
        action="store",
        type=int,
        default=1,
        help="the number of past steps to draw")
    parser.add_argument(
        "--carla-dt",
        action="store",
        type=float,
        default=0.1,
        help="time interval between two frames")
    parser.add_argument(
        "--carla-max-time-episode",
        action="store",
        type=int,
        default=1000,
        help="maximum time-step per episode")
    parser.add_argument(
        "--carla-max-waypt",
        action="store",
        type=int,
        default=12,
        help="maximum number of waypoints")
    parser.add_argument(
        "--carla-obs-range",
        action="store",
        type=int,
        default=32,
        help="observation range (meter)")
    parser.add_argument(
        "--carla-lidar-bin",
        action="store",
        type=float,
        default=0.125,
        help="bin size of lidar sensor (meter)")
    parser.add_argument(
        "--carla-d-behind",
        action="store",
        type=int,
        default=12,
        help="distance behind the ego vehicle (meter)")
    parser.add_argument(
        "--carla-out-lane-thres",
        action="store",
        type=float,
        default=2.0,
        help="threshold for out of lane")
    parser.add_argument(
        "--carla-desired-speed",
        action="store",
        type=float,
        default=8.0,
        help="desired speed (m/s)")
    parser.add_argument(
        "--carla-max-ego-spawn-times",
        action="store",
        type=int,
        default=100,
        help="maximum times to spawn ego vehicle")


    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail-algo', default='standard', help='algorithm to use: standard | wasserstein | none')
    parser.add_argument(
        '--expert-algo', default='ikostrikov', help='expert demos algorithm to use: a2c | ppo | acktr | ikostrikov')
    parser.add_argument(
        '--pretrain-algo', default='none', help='pre-training algorithm to use: cvae | bc | none')
    parser.add_argument(
        '--load-algo', default='a2c', help='algorithm to use: a2c | ppo | acktr | ikostrikov')
    parser.add_argument(
        '--test-model', default='trained', help='algorithm to use: trained | pretrained')
    
    
    parser.add_argument(
        '--latent-space', default='continuous', help='latent variable type to use: discrete | continuous')

    
    parser.add_argument(
        '--save-result',
        action='store_true',
        default=False,
        help='save results')
    parser.add_argument(
        '--render',
        action='store_true',
        default=False,
        help='render simulator')
    parser.add_argument(
        '--load-model',
        action='store_true',
        default=False,
        help='load trained model and obs_rms')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--task-transition',
        action='store_true',
        default=True,
        help='do imitation learning with task transition model')
    parser.add_argument(
        '--task-curiosity-reward',
        action='store_true',
        default=False,
        help='use task curiosity reward to train agent')
    parser.add_argument(
        '--reset-posterior',
        action='store_true',
        default=False,
        help='reset pre-trained posterior')
    parser.add_argument(
        '--reset-transition',
        action='store_true',
        default=False,
        help='reset pre-trained task transition model')
    parser.add_argument(
        '--fix-beta',
        action='store_true',
        default=False,
        help='fix beta to ignore regularization')
    parser.add_argument(
        '--good-end',
        action='store_true',
        default=False,
        help='tasks where end option is good transition')
    parser.add_argument(
        '--extract-obs',
        action='store_true',
        default=False,
        help='use some part of observation')
    parser.add_argument(
        '--use-latent',
        action='store_true',
        default=False,
        help='do imitation learning with latent variable')
    parser.add_argument(
        '--use-random-latent',
        action='store_true',
        default=False,
        help='do imitation learning with random latent variable')
    parser.add_argument(
        '--use-constant-latent',
        action='store_true',
        default=False,
        help='do imitation learning with constant latent variable')
    parser.add_argument(
        "--constant-latent",
        action="store",
        type=float,
        default=2.0,
        help="constant latent value")
    parser.add_argument(
        '--posterior',
        action='store_true',
        default=True,
        help='do imitation learning with posterior')
    parser.add_argument(
        '--hierarchical-policy',
        action='store_true',
        default=True,
        help='train policy with latent variable')
    parser.add_argument(
        "--init_beta",
        action="store",
        type=float,
        default=0,
        help="initial value of beta")
    parser.add_argument(
        "--i_c",
        action="store",
        type=float,
        default=0.2,
        help="value of information bottleneck")
    parser.add_argument(
        "--latent-dim",
        action="store",
        type=int,
        default=4,
        help="latent size for the generator")
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--posterior-batch-size',
        type=int,
        default=128,
        help='posterior batch size (default: 128)')
    parser.add_argument(
        '--recurrent-hidden-task-state-size',
        type=int,
        default=32,
        help='recurrent hidden state size for task transition model (default: 32)')
    parser.add_argument(
        '--episode', type=int, default=10, help='the number of result demonstrations  (default: 10')
    parser.add_argument(
        '--pretrain-epoch', type=int, default=700, help='number of pre-training epochs (default: 700)')
    parser.add_argument(
        '--bc-epoch', type=int, default=5, help='behavior cloning epochs (default:  5)')
    parser.add_argument(
        '--cvae-epoch', type=int, default=5, help='number of cvae epochs (default: 5)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=5, help='number of ppo epochs (default: 5)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--posterior-epoch', type=int, default=5, help='posterior epochs (default:  5)')
    parser.add_argument(
        '--task-epoch', type=int, default=5, help='task transition epochs (default:  5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for discriminator")
    parser.add_argument(
        "--p_lr", type=float, default=0.0001, help="learning rate for posterior")
    parser.add_argument(
        "--t_lr", type=float, default=0.0001, help="learning rate for task transition model")
    parser.add_argument(
        "--b_lr", type=float, default=0.0001, help="learning rate for beta in bottleneck")
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--posterior-reward-coef',
        type=float,
        default=0.1,
        help='posterior reward term coefficient (default: 0.1)')
    parser.add_argument(
        '--discr-reward-coef',
        type=float,
        default=0.1,
        help='discriminator reward term coefficient (default: 1.0)')
    parser.add_argument(
        '--task-reward-coef',
        type=float,
        default=0.1,
        help='task transition model reward term coefficient (default: 1.0)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--bc-loss-coef',
        type=float,
        default=0.5,
        help='behavior cloning loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=256,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--vis-interval',
        type=int,
        default=100,
        help='visualization interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=25,
        help='save interval, one save per n updates (default: 50)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        # default=None,
        help='eval interval, one eval per n updates (default: 1)')
    parser.add_argument(
        '--save-episode',
        type=int,
        default=100,
        help='eval episode, n episodes per one save (default: 100)')
    parser.add_argument(
        '--eval-episode',
        type=int,
        default=10,
        # default=None,
        help='eval episode, n episodes per one eval (default: 10)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        # default='/tmp/gym/',
        default='./logs/',
        help='directory to save learning logs (default: /tmp/gym)')
    parser.add_argument(
        '--pre-log-dir',
        # default='/tmp/gym/',
        default='./pre_logs/',
        help='directory to save learning logs (default: /tmp/gym)')
    parser.add_argument(
        '--result-dir',
        # default='/tmp/gym/',
        default='./results/',
        help='directory to save result logs (default: /tmp/gym)')
    parser.add_argument(
        '--pre-result-dir',
        # default='/tmp/gym/',
        default='./pre_results/',
        help='directory to save result logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent models (default: ./trained_models/)')
    parser.add_argument(
        '--experts-dir',
        default='./expert_models/',
        help='directory to save agent models (default: ./trained_models/)')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/',
        help='directory to load trained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--pre-load-dir',
        default='./pretrained_models/',
        help='directory to load pretrained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--pretrain-dir',
        default='./pretrained_models/',
        help='directory to save pretrained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--load-date',
        default='191025',
        help='pt file name for saved agent model')
    parser.add_argument(
        '--save-date',
        default='190909',
        help='pt file name for saved agent model')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    assert args.gail_algo in ['standard', 'wasserstein']

    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
