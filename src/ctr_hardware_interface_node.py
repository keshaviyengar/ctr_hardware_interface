#!/usr/bin/env python
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState
import tf2_ros

import serial
from serial_utils import *
from conversion_utils import *

# 250 steps_per_mm is cnc shield constant to convert for lead screws. Instead of changing on cnc shield, dividing here.

# Motor direction for translation is always 1
MOTOR_DIR = np.array([1.0, -1.0, -1.0])
DEG2STEPS = 16.0 / (1.8 * 250.0)  # micro_steps / (deg_per_step * step_per_mm)
MM2STEPS = 648.0 / (5.0 * 250.0)  # steps_per_rev / (mm_per_rev * micro_steps * step_per_mm)
HOME_OFFSET = np.array([-202.0, -94.0, 0.0])

NUM_TUBES = 3


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
        self.c_stepper = np.array(
            [648 * 1 / (5.0 * 250.0 * 16.0), 648 * 1 / (5.0 * 250.0 * 16.0), 648 * 1 / (5.0 * 250.0 * 16.0),
             # Translation
             16 * 1 / (1.8 * 250), 16 * 1 / (1.8 * 250),
             16 * 1 / (1.8 * 250)])  # Rotation, ctr motor signal unit conversion
        self.ctr_range = np.array([[-100, -6], [-90, -4], [-50, -2]])
        self.stepper_range = rel_beta2motor(self.ctr_range, MM2STEPS)
        self.velTr = 10000
        self.velRot = 5000
        self.tube_lengths = np.array([-300.0, -150.0, -60.0])

        # ROS-related
        self.tr_homing_service = rospy.Service("/translation_homing", Trigger, self.tr_homing_srv_callback)
        self.rot_homing_service = rospy.Service("/rotational_homing", Trigger, self.rot_homing_srv_callback)
        self.read_joint_state_service = rospy.Service("/read_joint_states", Trigger, self.read_joints_callback)
        self.joint_command_sub = rospy.Subscriber("/joint_command", JointState, self.joint_command_callback)
        self.joint_state_pub = rospy.Publisher("/joint_state", JointState, queue_size=10)
        # Setup a transform listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self._home_translation = False
        self._home_rotation = False
        self._read_joints = False
        self._command_joints = False
        self._last_command = None

        if not initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock):
            print("Translational homing failed...")
        # Test out conversion from motor2mm and mm2motor
        abs_betas, abs_alphas = self.read_joint_values()
        print("current beta: " + str(abs_betas))
        xyz_tr = abs_beta2motor(abs_betas, HOME_OFFSET, MM2STEPS)
        abs_beta_2 = motor2abs_beta(xyz_tr, HOME_OFFSET, 1 / MM2STEPS)
        # Check conversion is same both ways
        assert (np.array(abs_betas) == np.array(abs_beta_2)).all()

        print("Ready for commands!")
        self.controller_timer = rospy.Timer(rospy.Duration(0.05), self.controller_callback)

    def rot_homing_srv_callback(self, req):
        self._home_rotation = True
        return TriggerResponse(
            success=True,
            message="Successfully called rotational homing."
        )

    def tr_homing_srv_callback(self, req):
        self._home_translation = True
        return TriggerResponse(
            success=True,
            message="Successfully called translational homing."
        )

    def joint_command_callback(self, msg):
        self._last_command = np.array(msg.position)
        # Apply joint limits on extension
        self._command_joints = True

    def extension_limits(self, betas):
        for i in range(1, NUM_TUBES):
            # Ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            betas[i - 1] = min(betas[i - 1], betas[i])
            betas[i - 1] = max(betas[i - 1], self.tube_lengths[i] - self.tube_lengths[i - 1] + betas[i])
        return betas

    def read_joints_callback(self, req):
        self._read_joints = True
        return TriggerResponse(
            success=True,
            message="Successfully called to read joint states."
        )

    def publish_joint_state(self, betas, alphas):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['beta_0', 'beta_1', 'beta_2', 'alpha_0', 'alpha_1', 'alpha_2']
        msg.position = np.concatenate((betas, alphas))
        self.joint_state_pub.publish(msg)

    def read_joint_values(self):
        xyz_tr = motor_position(self.dev_tr, self.verbose)
        xyz_rot = motor_position(self.dev_rot, self.verbose)
        # betas, alphas = self.motor_to_joint(xyz_tr, xyz_rot)
        alphas = motor2alpha(xyz_rot, 1 / DEG2STEPS, MOTOR_DIR)
        betas = motor2abs_beta(xyz_tr, HOME_OFFSET, 1 / MM2STEPS)
        return betas, alphas

    def motor_to_joint(self, xyz_tr, xyz_rot):
        betas = np.multiply(xyz_tr, np.reciprocal(self.c_stepper[:3]))
        alphas_rel = np.multiply(np.reciprocal(self.c_stepper[3:]), xyz_rot)
        tmp = np.multiply(np.reciprocal(self.gtx_dir[3:]), alphas_rel)
        alphas = tmp * np.pi / 180.0
        return betas, alphas

    def joint_to_motor(self, betas, alphas):
        # Get relative betas as how done for carriage
        betas = np.flip(betas)
        alphas = np.flip(alphas) / np.pi * 180.0
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
        if self._home_rotation:
            print("rotational offsets: " + str(self.initialize_rot()))
            self._home_rotation = False
        # Read joints
        if self._read_joints:
            betas, alphas = self.read_joint_values()
            self.publish_joint_state(betas, alphas)
            self._read_joints = False
        if self._command_joints:
            xyz_tr_req = abs_beta2motor(self._last_command[:3], HOME_OFFSET, MM2STEPS)
            xyz_rot_req = alpha2motor(self._last_command[3:], DEG2STEPS, MOTOR_DIR)
            xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
            xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, True, self.verbose)
            betas = motor2abs_beta(xyz_tr_real, HOME_OFFSET, 1 / MM2STEPS)
            alphas = motor2alpha(xyz_rot_real, 1 / DEG2STEPS, MOTOR_DIR)
            self.publish_joint_state(betas, alphas)
            self._command_joints = False

    def calibrate_rot_offset(self, beta, tube, offsets):
        offset_rot = 0
        max_x = -np.inf
        rot_value = np.array([0.0, 0.0, 0.0]) + offsets
        for i in range(0, 180, 5):
            rot_value[tube] = float(i)
            xyz_tr_req, xyz_rot_req = self.joint_to_motor(beta, np.deg2rad(rot_value))
            xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
            # Get transform from tf frame for current tracker position
            try:
                trans = self.tfBuffer.lookup_transform("aurora_marker1", "base_link", rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("TF2 lookup failed...")
                continue
            if trans.transform.translation.x > max_x:
                max_x = trans.transform.translation.x
                print(max_x)
                offset_rot = i
            rospy.sleep(1.5)
        return offset_rot

    def initialize_rot(self):
        # Use configuration [-70.0, -30.0, -2.0]
        betas_rot_init = np.array([-70.0, -30.0, -2.0])
        xyz_tr_req, xyz_rot_req = self.joint_to_motor(betas_rot_init, np.array([0.0, 0.0, 0.0]))
        xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, True, self.verbose)
        xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
        rospy.sleep(5.0)
        offsets = np.array([0.0, 0.0, 0.0])
        for tube in range(2, -1, -1):
            print("offsets: ", str(offsets))
            offsets[tube] = self.calibrate_rot_offset(betas_rot_init, tube, offsets)
        # Move to zero position
        xyz_tr_req, xyz_rot_req = self.joint_to_motor(betas_rot_init, offsets)
        xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, False, self.verbose)
        xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
        return offsets


if __name__ == '__main__':
    rospy.init_node("ctr_hardware_interface_node")
    com_rot = "/dev/arduino_controller"
    com_tr = "/dev/qinheng_controller"
    ctr_hw_interface = CTRHardwareInterface(com_rot, com_tr, is_lock=True)
    rospy.spin()
