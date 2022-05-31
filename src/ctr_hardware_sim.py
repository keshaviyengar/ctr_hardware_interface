import rospy
import rospkg

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

#from stable_baselines.her.utils import HERGoalEnvWrapper
import numpy as np
import gym
import ctr_reach_envs
from ctr_reach_envs.envs.obs_utils import rep2joint, ego2prop

NUM_TUBES = 3


# Hardware simulation class for testing the the policy node
# Receives joint_command values, applies to environment and returns new observation


class ConcentricTubeRobotSimNode(object):
    def __init__(self, env_id):
        # TODO: Add in robot tube parameters
        env_kwargs = {
            'ctr_systems_parameters': {
                'ctr_0': {
                    'tube_0':
                        {'length': 304e-3, 'length_curved': 193e-3, 'diameter_inner': 0.49e-3, 'diameter_outer': 1.10e-3,
                         'stiffness': 75e+10, 'torsional_stiffness': 25e+10, 'x_curvature': 19.5, 'y_curvature': 0
                         },
                    'tube_1':
                        {'length': 153e-3, 'length_curved': 73e-3, 'diameter_inner': 1.12e-3, 'diameter_outer': 1.34e-3,
                         'stiffness': 75e+10, 'torsional_stiffness': 25e+10, 'x_curvature': 16.23, 'y_curvature': 0
                         },
                    'tube_2':
                        {'length': 68e-3, 'length_curved': 68e-3, 'diameter_inner': 1.36e-3, 'diameter_outer': 1.82e-3,
                         'stiffness': 75e+10, 'torsional_stiffness': 25e+10, 'x_curvature': 14.86, 'y_curvature': 0
                         }
                }
            },
            'extension_action_limit': 0.001,
            'rotation_action_limit': 5,
            'max_steps_per_episode': 150,
            'n_substeps': 10,
            'goal_tolerance_parameters': {
                'inc_tol_obs': False, 'final_tol': 0.001, 'initial_tol': 0.020,
                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001
            },
            'noise_parameters': {
                # 0.001 is the gear ratio
                # 0.001 is also the tracking std deviation for now for testing.
                'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
            },
            'select_systems': [0],
            'constrain_alpha': False,
            # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
            'initial_joints': np.array([0, 0, 0, 0, 0, 0]),
            'joint_representation': 'egocentric',
            'resample_joints': False,
            'evaluation': True,
            'length_based_sample': False,
            'domain_rand': 0.0
        }

        self.env = gym.make((env_id), **env_kwargs)
        self.goal_tolerance = 0.001

        self.desired_goal = None

        # Subscribe to actions
        self.joint_action_sub = rospy.Subscriber("/joint_action", JointState, self.joint_action_callback)
        self.desired_sub = rospy.Subscriber("/desired_goal", JointState, self.desired_goal_callback)

        # Publish joints, tip pos, desired_pos
        self.joint_states_pub = rospy.Publisher("/joint_state", JointState, queue_size=10)
        self.achieved_goal_pub = rospy.Publisher("/ndi/achieved_goal", PoseStamped, queue_size=10)

        # Keep track of joint values for timer based publishing
        self.read_joints_timer = rospy.Timer(rospy.Duration(0.5), self.read_joints_callback)
        self.obs = self.env.reset()

    def desired_goal_callback(self, msg):
        self.desired_goal = np.array([msg.position.x, msg.position.y, msg.position.z])

    def joint_action_callback(self, msg):
        joint_action = msg.position
        obs, reward, done, info = self.env.step(joint_action)
        self.obs = obs

    def read_joints_callback(self, event):
        betas, alphas = self.read_joint_values()
        msg = self.create_joint_msg(np.concatenate((betas, alphas)))
        self.joint_states_pub.publish(msg)
        self.publish_tip_position()

    def read_joint_values(self):
        # Get the trigonometric representation in observation, convert to regular proprioceptive,
        # basic joint representation
        joint_values = ego2prop(rep2joint(self.obs["observation"][:3 * NUM_TUBES]))
        betas = joint_values[:3]
        alphas = joint_values[3:]
        return betas, alphas

    def publish_tip_position(self):
        # Tip position
        achieved_goal_msg = PoseStamped()
        achieved_goal_msg.header.stamp = rospy.Time.now()
        achieved_goal_msg.pose.position.x = self.obs["achieved_goal"][0]
        achieved_goal_msg.pose.position.y = self.obs["achieved_goal"][1]
        achieved_goal_msg.pose.position.z = self.obs["achieved_goal"][2]
        self.achieved_goal_pub.publish(achieved_goal_msg)

    @staticmethod
    def create_joint_msg(joint_values):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        # Ordered 0 (innermost), 1 (middle), 2 (outer)
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        betas = joint_values[:NUM_TUBES]
        alphas = joint_values[NUM_TUBES:]
        msg.position = np.concatenate((betas, alphas))
        return msg


if __name__ == '__main__':
    rospy.init_node("ctr_sim")
    ctr_sim = ConcentricTubeRobotSimNode("CTR-Reach-v0")
    rospy.spin()
