#!/usr/bin/env python
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState
import tf2_ros

import serial
from serial_utils import *
from conversion_utils import *
from CTR_Python.Tube import Tube

# 250 steps_per_mm is cnc shield constant to convert for lead screws. Instead of changing on cnc shield, dividing here.

# Motor direction for translation is always 1
MOTOR_DIR = np.array([-1.0, -1.0, -1.0])
DEG2STEPS = 16.0 / (1.8 * 250.0)  # micro_steps / (deg_per_step * step_per_mm)
MM2STEPS = 200.0 * 4.0 / (2.0 * 4) * 1 / 250.0  # steps_per_rev * micro_steps / (mm_per_rev) # Divide by 250 to cancel out default setting
ROTATION_OFFSET = np.array([])  # offset from rotational homing

NUM_TUBES = 3


class CTRHardwareInterface(object):
    def __init__(self, com_rot, com_tr, is_lock=False):
        self.dev_rot = serial.Serial(port=com_rot, baudrate=115200, timeout=10.0)
        self.dev_tr = serial.Serial(port=com_tr, baudrate=115200, timeout=10.0)
        self.is_lock = is_lock
        self.verbose = False

        # Motor parameters
        self.gtx_dir = np.array([1, 1, 1,  # Translation
                                 -1, -1, -1])  # Rotation
        self.ctr_range = np.array([[-97, 0], [-76, -0], [-47, 0]])
        self.stepper_range = beta2motor(self.ctr_range, MM2STEPS)
        print("ctr_range: " + str(self.ctr_range))
        print("stepper_range: " + str(self.stepper_range))
        self.velTr = 10000
        self.velRot = 5000
        # Load ctr robot parameters
        # Get parameters from ROS params
        tube_0 = Tube(**rospy.get_param("/tube_0"))
        tube_1 = Tube(**rospy.get_param("/tube_1"))
        tube_2 = Tube(**rospy.get_param("/tube_2"))

        #self.tube_lengths = np.array([tube_0.L, tube_1.L, tube_2.L])

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

        cnc_initialized = False
        while not cnc_initialized:
            cnc_initialized = initialize_cnc(self.dev_rot, self.dev_tr, self.verbose, self.is_lock)
            if cnc_initialized:
                break
            print("Translational homing failed... retrying")

        # Test out conversion from motor2mm and mm2motor
        abs_betas, abs_alphas = self.read_joint_values()
        xyz_tr = beta2motor(abs_betas, MM2STEPS)
        abs_beta_2 = motor2beta(xyz_tr, 1 / MM2STEPS)
        # Check conversion is same both ways
        assert (np.array(abs_betas) == np.array(abs_beta_2)).all()
        print("Starting joint values: ")
        print("betas: " + str(abs_betas))
        print("alphas: " + str(abs_alphas))

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
        self._last_command[:3] = self.extension_limits(self._last_command[:3])
        self._command_joints = True

    def extension_limits(self, betas):
        #betas = np.flip(betas)
        #for i in range(1, NUM_TUBES):
        #    # Ordering is reversed, since we have innermost as last whereas in constraints its first.
        #    # Bi >= Bi-1
        #    # Bi <= Bi-1 + Li-1 - Li
        #    betas[i-1] = max(betas[i-1], betas[i])
        #    betas[i-1] = min(betas[i-1], self.tube_lengths[i] * 1000 - self.tube_lengths[i-1] * 1000 + betas[i])
        #betas = np.flip(betas)
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
        alphas = motor2alpha(xyz_rot, 1 / DEG2STEPS, MOTOR_DIR)
        betas = motor2beta(xyz_tr, 1 / MM2STEPS)
        return betas, alphas

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
            xyz_tr_req = beta2motor(self._last_command[:3], MM2STEPS)
            xyz_rot_req = alpha2motor(self._last_command[3:], DEG2STEPS, MOTOR_DIR)
            xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
            xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, True, self.verbose)
            betas = motor2beta(xyz_tr_real, 1 / MM2STEPS)
            alphas = motor2alpha(xyz_rot_real, 1 / DEG2STEPS, MOTOR_DIR)
            self.publish_joint_state(betas, alphas)
            self._command_joints = False

    def initialize_rot(self):
        # Homing beta configurations, retract tubes to have tracker at tip of current tube.
        # Ordering outer to inner
        betas_rot_homing = np.array([[-1.0, -1.0, -1.0], [-54.0, -1.0, -1.0], [-82.0, -27.0, -1.0]])
        # Run rotational homing and set zero position
        alpha_range = 10.0
        min_y_val = np.array([np.inf, np.inf, np.inf])
        zero_joints = np.zeros(3)
        rospy.sleep(5.0)
        rospy.loginfo("waiting 5 seconds...")
        for i in range(NUM_TUBES-1, -1, -1):
            print("Tube " + str(i) + " rotation.")
            xyz_tr_req = beta2motor(betas_rot_homing[i], MM2STEPS)
            xyz_tr_real = safe_move(self.dev_tr, xyz_tr_req, self.velTr, self.stepper_range, True, self.verbose)
            betas = motor2beta(xyz_tr_real, 1 / MM2STEPS)
            rospy.sleep(2.0)
            for j in np.linspace(-alpha_range, alpha_range, 30):
                rot_command = np.zeros(3)
                rot_command[i] = j
                xyz_rot_req = alpha2motor(rot_command, DEG2STEPS, MOTOR_DIR)
                xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
                # Get five measurements and average
                num_samples = 5
                y_samples = np.zeros(num_samples)
                for k in range(num_samples):
                    try:
                        trans = self.tfBuffer.lookup_transform("entry_point", "aurora_marker1", rospy.Time())
                        y_samples[k] = trans.transform.translation.y
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rospy.logerr("TF2 lookup failed...")
                        continue
                    rospy.sleep(1.0 / num_samples)
                rospy.loginfo("mean y_values: " + str(np.mean(np.abs(y_samples))))
                if np.mean(np.abs(y_samples)) <= min_y_val[i]:
                    min_y_val[i] = np.mean(np.abs(y_samples))
                    print("New alignment!: " + str(min_y_val))
                    zero_joints[i] = j
            rospy.loginfo("zero_joints" + str(zero_joints))
            rospy.loginfo("Moving to new home position...")
            xyz_rot_req = alpha2motor(zero_joints, DEG2STEPS, MOTOR_DIR)
            xyz_rot_real = safe_move(self.dev_rot, xyz_rot_req, self.velRot, self.stepper_range, False, self.verbose)
            rospy.sleep(2.0)
        rospy.loginfo('min_x_values: ' + str(min_y_val))
        return zero_joints


if __name__ == '__main__':
    rospy.init_node("ctr_hardware_interface_node")
    com_rot = "/dev/arduino_controller"
    com_tr = "/dev/qinheng_controller"
    ctr_hw_interface = CTRHardwareInterface(com_rot, com_tr, is_lock=True)
    rospy.spin()
