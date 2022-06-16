#!/usr/bin/env python
import rospy
import rospkg

import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

import numpy as np
from ctr_model import CTRModel
from CTR_Python.Tube import Tube

NUM_TUBES = 3


# Hardware simulation class for testing the the policy node
# Can set in either standalone mode, where returns current joint values and tip position
# Or can be used in conjunction with hardware interface node to compare model to real world


class ConcentricTubeRobotSimNode(object):
    def __init__(self, standalone=False):
        self.standalone = standalone
        # Get parameters from ROS params
        tube_0 = Tube(**rospy.get_param("/tube_0"))
        tube_1 = Tube(**rospy.get_param("/tube_1"))
        tube_2 = Tube(**rospy.get_param("/tube_2"))

        ctr_parameters = [tube_0, tube_1, tube_2]
        # Create the FK model object
        self.fk_model = CTRModel(ctr_parameters)

        # Subscribers, publishers and broadcasters
        if standalone:
            self.read_joint_state_service = rospy.Service("/read_joint_states", Trigger, self.read_joints_callback)
            self.joint_state_pub = rospy.Publisher("/joint_state", JointState, queue_size=10)
            self.joint_command_sub = rospy.Subscriber("/joint_command", JointState, self.joint_command_callback)
            self.alphas = np.array([0, 0, 0])
            self.betas = np.array([0, 0, 0])
            self.init_joints = np.array([0, 0, 0, 0, 0, 0])
        else:
            print("waiting for read_joints service...")
            rospy.wait_for_service("/read_joint_states")
            self.joint_state_sub = rospy.Subscriber("/joint_state", JointState, self.joint_state_callback)
        self.controller_timer = rospy.Timer(rospy.Duration(0.05), self.controller_callback)
        self.br = tf2_ros.TransformBroadcaster()

        self.desired_goal = None
        self.alphas = None
        self.betas = None
        self._read_joints = False
        self._command_joints = False
        self._last_command = None

        read_joint_state_service = rospy.ServiceProxy("/read_joint_states", Trigger)

        resp = read_joint_state_service(TriggerRequest())
        print("Trigger read joints.")

    def joint_command_callback(self, msg):
        self._last_command = np.array(msg.position)
        # Apply joint limits on extension
        self._last_command[3:] = self.extension_limits(self._last_command[3:])
        self._command_joints = True

    def extension_limits(self, betas):
        tube_lengths = [self.fk_model.ctr_parameters[0].Length, self.fk_model.ctr_parameters[1].Length,
                        self.fk_model.ctr_parameters[2].Length]
        for i in range(1, NUM_TUBES):
            # Ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            betas[i - 1] = min(betas[i - 1], betas[i])
            betas[i - 1] = max(betas[i - 1], tube_lengths[i] - tube_lengths[i - 1] + betas[i])
        return betas

    def read_joints_callback(self, req):
        self._read_joints = True
        return TriggerResponse(
            success=True,
            message="Successfully called to read joint states."
        )

    def joint_state_callback(self, msg):
        self.betas = np.array(msg.position[:3]) / 1000.0
        self.alphas = np.array(msg.position[3:])

    def broadcast_tip_position(self, tip_pose):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = "sim_marker"
        t.transform.translation.x = tip_pose[0]
        t.transform.translation.y = tip_pose[1]
        t.transform.translation.z = tip_pose[2]
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1
        self.br.sendTransform(t)

    def publish_joint_state(self, betas, alphas):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        msg.position = np.concatenate((betas, alphas))
        self.joint_state_pub.publish(msg)

    # Callback for all controller communication to avoid overloading ports
    def controller_callback(self, event):
        # Read joints
        if self._read_joints:
            if self.betas is not None and self.alphas is not None:
                self.publish_joint_state(self.betas, self.alphas)
            self._read_joints = False
        if self._command_joints:
            self.publish_joint_state(self.betas, self.alphas)
            self._command_joints = False
        if self.betas is not None and self.alphas is not None:
            joints = np.concatenate((self.betas, self.alphas))
            # Waiting for betas and alphas to be obtained
            # Get current tip position
            tip_pos = self.fk_model.forward_kinematics(joints)
            self.broadcast_tip_position(tip_pos)


if __name__ == '__main__':
    rospy.init_node("ctr_sim")
    ctr_sim = ConcentricTubeRobotSimNode()
    rospy.spin()
