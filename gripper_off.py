import time
from piper_sdk import C_PiperInterface_V3

piper = C_PiperInterface_V3("can_left")
piper.ConnectPort()
time.sleep(1.0)

# Disable only the gripper. No arm enable/reset or zero-setting.
for _ in range(10):
    piper.GripperCtrl_V3(
        gripper_val=0,
        gripper_effort=0,
        gripper_code=0x00,
        set_zero=0x00,
    )
    time.sleep(0.05)

for _ in range(5):
    print(piper.GetArmGripperMsgs())
    time.sleep(0.2)