import unittest

import aubo_workbench.gripper_control as module


class GripperControlModuleTests(unittest.TestCase):
    def test_driver_path_exists(self):
        self.assertTrue(module.DRIVER_PATH.exists(), module.DRIVER_PATH)

    def test_driver_loads_without_opening_serial_port(self):
        driver = module.load_gripper_driver()
        gripper = driver.ZErg20C(
            port="COM_TEST",
            slave_id=1,
            baudrate=115200,
            auto_sync_endianness=False,
        )
        self.assertFalse(gripper.is_connected())


if __name__ == "__main__":
    unittest.main()
