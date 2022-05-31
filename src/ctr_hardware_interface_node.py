#!/usr/bin/env python
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState

import serial
from serial_utils import *


class CTRHardwareInterface(object):
    def __init__(self, com_rot, com_tr, is_lock=False):
        # If to printout inputs and outputs to controller for debugging
        connected = False
        while not connected:
            try:
                self.dev_rot = serial.Serial(port=com_rot, baudrate=115200, timeout=10.0)
                self.dev_tr = serial.Serial(port=com_tr, baudrate=115200, timeout=10.0)
                connected = True
            except serial.serialutil.SerialException:
                connected = False
                rospy.sleep(2.0)
                print("Waiting for connection with controllers...")
        self.is_lock = is_lock
        self.verbose = False

        # Motor parameters
        self.gtx_dir = np.array([1, 1, 1,  # Translation
                                 1, -1, -1])  # Rotation
        self.c_stepper = np.array([1 / 2.5, 1 / 2.5, 1 / 2.5,  # Translation
                                   13 * 3 * 1.3, 13 * 1.3, 13 * 1.3])  # Rotation, ctr motor signal unit conversion
        self.ctr_range = np.array([[-50, -2], [-90, -4], [-100, -6]])
        self.stepper_range = np.multiply(self.ctr_range, self.c_stepper[:2])
        self.velTr = 10000
        self.velRot = 1000

        # ROS-related
        self.homing_service = rospy.Service("/translation_homing", Trigger, self.homing_srv_callback)
        self.joint_command_sub = rospy.Subscriber("/joint_command", JointState, self.joint_command_callback)
        self.joint_state_pub = rospy.Publisher("/joint_state", JointState, queue_size=10)

        self._home_translation = False
        self._read_joints = False
        self._command_joints = False
        self._last_command = None

        if not initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock):
            print("Translational homing failed...")

        self.controller_timer = rospy.Timer(rospy.Duration(0.05), self.controller_callback)
        # self.read_joints_timer = rospy.Timer(rospy.Duration(0.1), self.read_joints_callback)

    def homing_srv_callback(self, req):
        self._home_translation = True

    def joint_command_callback(self, msg):
        self._last_command = msg
        self._command_joints = True

    def read_joints_callback(self, event):
        self._read_joints = True

    def read_joint_values(self):
        xyz_tr = motor_position(self.dev_tr, self.verbose)
        xyz_rot = motor_position(self.dev_rot, self.verbose)
        betas, alphas = self.motor_to_joint(xyz_tr, xyz_rot)
        return betas, alphas

    def motor_to_joint(self, xyz_tr, xyz_rot):
        betas = np.multiply(xyz_tr, np.reciprocal(self.c_stepper[:3]))
        tmp = np.multiply(np.reciprocal(self.gtx_dir[3:]), xyz_rot)
        alphas = tmp * np.pi * 2
        return betas, alphas

    def joint_to_motor(self, betas, alphas):
        # Flip the ordering
        betas = np.flip(betas)
        alphas = np.flip(alphas / np.pi / 2)
        xyz_tr = np.multiply(betas, self.c_stepper[:3])
        tmp = np.multiply(self.gtx_dir[3:], alphas)
        xyz_rot = np.multiply(tmp, self.c_stepper[3:])
        return xyz_tr, xyz_rot

    # Callback for all controller communication to avoid overloading ports
    def controller_callback(self, event):
        # Home translation
        if self._home_translation:
            if initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock):
                rospy.loginfo("homing completed.")
            else:
                rospy.loginfo("homing failed.")
        # Read joints
        if self._read_joints:
            betas, alphas = self.read_joint_values()
            self.publish_joint_state(betas, alphas)
            self._read_joints = False
        if self._command_joints:
            xyz_tr_req, xyz_rot_req = self.joint_to_motor(np.array(self._last_command.position[:3]),
                                                          np.array(self._last_command.position[3:]))
            xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, True, self.verbose)
            xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, True, self.verbose)
            betas, alphas = self.motor_to_joint(xyz_tr_real, xyz_rot_real)
            self.publish_joint_state(betas, alphas)
            self._command_joints = False

    def publish_joint_state(self, betas, alphas):
        # Motor values read will be in opposite order so flip
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        msg.position = np.concatenate((np.flip(betas), np.flip(alphas)))
        self.joint_state_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("ctr_hardware_interface_node")
    com_rot = "/dev/ttyACM0"
    com_tr = "/dev/ttyUSB0"
    ctr_hw_interface = CTRHardwareInterface("/dev/ttyACM0", "/dev/ttyUSB0", is_lock=True)
    rospy.spin()
