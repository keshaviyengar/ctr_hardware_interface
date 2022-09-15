#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
import tf2_ros

from conversion_utils import *
import pandas as pd
import os
from datetime import datetime

# ROS Image message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


# Conversion between normalized and un-normalized joints
def B_U_to_B(B_U, L_1, L_2, L_3):
    B_U = np.append(B_U, 1)
    M_B = np.array([[-L_1, 0, 0],
                    [-L_1, L_1 - L_2, 0],
                    [-L_1, L_1 - L_2, L_2 - L_3]])
    normalized_B = np.block([[0.5 * M_B, 0.5 * np.matmul(M_B, np.ones((3, 1)))],
                             [np.zeros((1, 3)), 1]])
    B = np.matmul(normalized_B, B_U)
    return B[:3]


def B_to_B_U(B, L_1, L_2, L_3):
    B = np.append(B, 1)
    M_B = np.array([[-L_1, 0, 0],
                    [-L_1, L_1 - L_2, 0],
                    [-L_1, L_1 - L_2, L_2 - L_3]])
    normalized_B = np.block([[0.5 * M_B, 0.5 * np.matmul(M_B, np.ones((3, 1)))],
                             [np.zeros((1, 3)), 1]])
    B_U = np.matmul(np.linalg.inv(normalized_B), B)
    return B_U[:3]


def alpha_U_to_alpha(alpha_U, alpha_max):
    return alpha_max * alpha_U


def alpha_to_alpha_U(alpha, alpha_max):
    return 1 / alpha_max * alpha


class WorkspaceCollection(object):
    def __init__(self, n_samples, alpha_max, image_topic, log_path='data/'):
        self.n_samples = n_samples
        self.alpha_max = alpha_max
        self.log_path = log_path
        # Publisher node
        self.joint_command_pub = rospy.Publisher("/joint_command", JointState, queue_size=10)
        # Setup a transform listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.global_df = pd.DataFrame(columns=["Timestamp", "beta_0", "beta_1", "beta_2",
                                               "alpha_0", "alpha_1", "alpha_2",
                                               "del_beta_0", "del_beta_1", "del_beta_2",
                                               "del_alpha_0", "del_alpha_1", "del_alpha_2",
                                               "x", "y", "z", "e_1", "e_2", "e_3", "n_1"
                                               ])

        self.bridge = CvBridge()
        self.cv2_img = None
        self.cam_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

    # Ordering for this is reversed outermost to innermost
    def sorted_sampling(self, u_beta, u_alpha, L_star, w):
        # L_star is margin included lengths
        # Sample alphas
        alphas_U = np.random.uniform(low=-np.ones((self.n_samples, 3)), high=np.ones((self.n_samples, 3)))
        alphas = alpha_U_to_alpha(alphas_U, self.alpha_max)
        # Sample betas
        B_U = np.random.uniform(low=-np.ones((self.n_samples, 3)), high=np.ones((self.n_samples, 3)))
        B_i = np.zeros_like(B_U)
        D_Ts = np.inf * np.ones(self.n_samples)
        for n in range(self.n_samples):
            D_T = 0
            B_i[n, :] = B_U_to_B(B_U[n, :], L_star[0], L_star[1], L_star[2])
            for i in range(3):
                D_T += w[i] * u_alpha * (alphas[n, i] - self.alpha_max) - w[2 + i] * u_beta * B_i[n, i]
            D_Ts[n] = D_T

        sorted_order = np.argsort(D_Ts)
        B_sorted = B_i[sorted_order, :]
        alphas_sorted = alphas[sorted_order, :]
        return B_sorted, alphas_sorted, sorted_order

    # Sample along a single plane (all tubes rotate together)
    def planar_sampling(self, alpha_max, alpha_step):
        alpha_values = np.linspace(-alpha_max, alpha_max, alpha_step)
        alpha_planar = np.repeat(alpha_values, 3).reshape((alpha_step, 3))
        B_planar = np.zeros((alpha_step, 3))
        return B_planar, alpha_planar

    # Rotate only one tube
    def tube_rotate_sampling(self, alpha_max, alpha_step, tube):
        alpha_values = np.linspace(-alpha_max, alpha_max, alpha_step)
        alpha_rotate = np.zeros((alpha_step, 3))
        alpha_rotate[:, tube] = alpha_values
        B_rotate = np.zeros((alpha_step, 3))
        return B_rotate, alpha_rotate

    # Extend in follow-the-leader approach. Extend outer first, extend middle finally extend inner
    def tube_extend_sampling(self, beta_max, beta_step):
        L_margin = np.array([136.5 - 15.0, 77.0 - 5.0, 47.5])
        # Fully retracted position
        retracted = -L_margin
        # Fully extend outer tube
        beta_extend_2 = np.array([np.linspace(retracted[0], retracted[0] - retracted[2], int(beta_step / 3)),
                                  np.linspace(retracted[1], retracted[1] - retracted[2], int(beta_step / 3)),
                                  np.linspace(retracted[2], beta_max, int(beta_step / 3)),
                                  ])
        # Fully extend middle tube
        beta_extend_1 = np.array([np.linspace(retracted[0] - retracted[2], retracted[0] - retracted[1], int(beta_step / 3)),
                                  np.linspace(retracted[1] - retracted[2], -1.0, int(beta_step / 3)),
                                  np.linspace(beta_max, beta_max, int(beta_step / 3))
                                  ])
        # Fully extend inner tube
        beta_extend_0 = np.array([np.linspace(retracted[0] - retracted[1], beta_max, int(beta_step / 3)),
                                  np.linspace(beta_max, beta_max, int(beta_step / 3)),
                                  np.linspace(beta_max, beta_max, int(beta_step / 3))
                                  ])
        betas_extend = np.concatenate((np.concatenate((beta_extend_2, beta_extend_1), axis=1), beta_extend_0), axis=1)
        alphas_extend = np.zeros_like(betas_extend)
        return betas_extend.T, alphas_extend.T

    def explore_workspace(self, betas, alphas):
        # Move to first position and wait
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        msg.position = np.concatenate((betas[0, :], alphas[0, :]))
        rospy.loginfo(msg.position)
        self.joint_command_pub.publish(msg)
        rospy.sleep(20.0)
        count = 1
        total_count = len(betas) - 1
        for prev_beta, prev_alpha, beta, alpha in zip(betas, alphas, betas[1:], alphas[1:]):
            print("current sample: " + str(count) + ' / ' + str(total_count))
            count = count + 1
            local_df = pd.DataFrame(columns=["Timestamp", "beta_0", "beta_1", "beta_2",
                                             "alpha_0", "alpha_1", "alpha_2",
                                             "del_beta_0", "del_beta_1", "del_beta_2",
                                             "del_alpha_0", "del_alpha_1", "del_alpha_2",
                                             "x", "y", "z", "e_1", "e_2", "e_3", "n_1"
                                             ])
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
            # msg.position = np.concatenate((np.flip(beta), np.flip(alpha)))
            msg.position = np.concatenate((beta, alpha))
            rospy.loginfo(msg.position)
            self.joint_command_pub.publish(msg)
            # Allow robot 1 second to reach new point
            rospy.sleep(1.0)
            # Allow 0.2 seconds between measurements
            samples_per_measurement = 5
            translations = []
            rotations = []
            for k in range(samples_per_measurement):
                try:
                    ee_pose = self.tfBuffer.lookup_transform("entry_point", "aurora_marker1", rospy.Time())
                    translation = [ee_pose.transform.translation.x, ee_pose.transform.translation.y,
                                   ee_pose.transform.translation.z]
                    rotation = [ee_pose.transform.rotation.x, ee_pose.transform.rotation.y,
                                ee_pose.transform.rotation.z, ee_pose.transform.rotation.w]
                    translations.append(translation)
                    rotations.append(rotation)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logerr("TF2 lookup failed...")
                rospy.sleep(2.0 / samples_per_measurement)
            translations = np.array(translations)
            rotations = np.array(rotations)
            new_row = {'Timestamp': msg.header.stamp,
                       'beta_0': beta[0], 'beta_1': beta[1], 'beta_2': beta[2],
                       'alpha_0': alpha[0], 'alpha_1': alpha[1], 'alpha_2': alpha[2],
                       'del_beta_0': beta[0] - prev_beta[0], 'del_beta_1': beta[1] - prev_beta[1],
                       'del_beta_2': beta[2] - prev_beta[2],
                       'del_alpha_0': alpha[0], 'del_alpha_1': alpha[1], 'del_alpha_2': alpha[2],
                       'x': np.mean(translations[:, 0]), 'y': np.mean(translations[:, 1]),
                       'z': np.mean(translations[:, 2]),
                       'e_1': np.mean(rotations[:, 0]), 'e_2': np.mean(rotations[:, 1]),
                       'e_3': np.mean(rotations[:, 2]),
                       'n_1': np.mean(rotations[:, 3])
                       }
            local_df = local_df.append(new_row, ignore_index=True)

            def get_position_file_name(position):
                pos_str = ''
                for i in np.around(position, decimals=2):
                    pos_str += str(i) + '_'
                return pos_str[:-1]

            if self.cv2_img is not None:
                cv2.imwrite(self.log_path + get_position_file_name(msg.position) + '.jpeg', self.cv2_img)
            # Create / open csv file of name joint_values
            if os.path.exists(self.log_path + get_position_file_name(msg.position) + '.csv'):
                # open and append
                loaded_df = pd.read_csv(self.log_path + get_position_file_name(msg.position) + '.csv')
                loaded_df.append(local_df, ignore_index=True)
                loaded_df.to_csv(self.log_path + get_position_file_name(msg.position) + '.csv')
            else:
                # create file
                local_df.to_csv(self.log_path + get_position_file_name(msg.position) + '.csv')

            # Write num_samples with all other data to global csv file
            self.global_df = self.global_df.append(local_df, ignore_index=True)
            self.global_df.to_csv(self.log_path + 'all_data.csv')
        # Return to home position
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        msg.position = [-15.0, -10.0, -5.0, 0, 0, 0]
        self.joint_command_pub.publish(msg)

    def image_callback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV2
            self.cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)


if __name__ == '__main__':
    rospy.init_node("workspace_collection")
    # Transmission for systems
    u_alpha = 16.0 / (1.8 * 250.0)  # micro_steps / (deg_per_step * step_per_mm)
    u_beta = 200.0 * 4.0 / (
            2.0 * 4) * 1 / 250.0  # steps_per_rev * micro_steps / (mm_per_rev) # Divide by 250 to cancel out default setting
    # weights for distance
    w = np.array([0.1, 0.1, 0.1, 0.4, 0.2, 0.1])
    # Number of samples
    n_samples = 200
    alpha_max = np.rad2deg(np.pi / 3)
    # Length of tubes, considering a margin of 5mm for innermost tube. Ordered outermost to innermost
    L_star = np.array([47.5, 77.0 - 5.0, 136.5 - 15.0])
    image_topic = "usb_cam/image_raw"
    sample_method = 'random'
    # Create log path
    log_path = sample_method + '/' + datetime.now().strftime("%d_%m_%y_%H%M") + "/"
    os.mkdir(log_path)
    workspace_collector = WorkspaceCollection(n_samples, alpha_max, image_topic, log_path)
    # Sample based on method selected
    if sample_method == 'random':
        B_sorted, alphas_sorted, sorted_order = workspace_collector.sorted_sampling(u_beta, u_alpha, L_star, w)
        # Based on random sampling, collect shortest path to all joint values
        workspace_collector.explore_workspace(np.flip(B_sorted), np.flip(alphas_sorted))
    elif sample_method == 'planar':
        B_planar, alphas_planar = workspace_collector.planar_sampling(alpha_max, n_samples)
        workspace_collector.explore_workspace(B_planar, alphas_planar)
    elif sample_method == 'single_tube_rotation':
        B_single_tube, alphas_single_tube = workspace_collector.tube_rotate_sampling(alpha_max, n_samples, tube=0)
        workspace_collector.explore_workspace(B_single_tube, alphas_single_tube)
    elif sample_method == 'extension':
        B_single_extend, alphas_single_extend = workspace_collector.tube_extend_sampling(-2.0, 60)
        workspace_collector.explore_workspace(B_single_extend, alphas_single_extend)
    else:
        rospy.loginfo("Incorrect sample method chosen...")
    rospy.spin()
