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
        self.ctr_range = np.array([[-50, 2], [-90, -4], [-100, 6]])
        self.stepper_range = np.multiply(self.ctr_range, self.c_stepper[:2])
        self.gtx_range = [0.12, -0.05]
        self.velTr = 10000
        self.velRot = 1000

        # ROS-related
        self.homing_service = rospy.Service("/translation_homing", Trigger, self.homing_srv_callback)
        self.joint_command_sub = rospy.Subscriber("/joint_command", JointState, self.joint_command_callback)
        self.joint_state_pub = rospy.Publisher("/joint_state", JointState, queue_size=10)

        if not initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock):
            print("Translational homing failed...")

    def homing_srv_callback(self, req):
        if initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock):
            rospy.sleep(5.0)
            return TriggerResponse(success=True, message="Homing completed.")

    def joint_command_callback(self, msg):
        betas = np.array(msg.position[:3])
        alphas = np.array(msg.position[3:]) / np.pi / 2

        xyz_tr_request = np.multiply(betas, self.c_stepper[:3])
        xyz_tr_real = safe_move(self.dev_tr, xyz_tr_request, self.velTr, self.stepper_range, True, self.verbose)
        tmp = np.multiply(self.gtx_dir[3:], alphas)
        xyz_rot_request = np.multiply(tmp, self.c_stepper[3:])
        xyz_rot_real = safe_move(self.dev_rot, xyz_rot_request, self.velRot, self.stepper_range, True, self.verbose)


if __name__ == '__main__':
    rospy.init_node("ctr_hardware_interface_node")
    com_rot = "/dev/ttyACM0"
    com_tr = "/dev/ttyUSB0"
    ctr_hw_interface = CTRHardwareInterface("/dev/ttyACM0", "/dev/ttyUSB0", is_lock=True)
    rospy.spin()
