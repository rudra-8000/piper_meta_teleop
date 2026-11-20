import time
from piper_sdk import C_PiperInterface_V2, C_PiperInterface_V3

piper = C_PiperInterface_V3("can_left")
print("Interface class:", type(piper).__name__)
piper.ConnectPort()
time.sleep(1.0)
import inspect

print("GripperCtrl signature:", inspect.signature(piper.GripperCtrl))
print(inspect.getsource(piper.GripperCtrl))

piper.SearchPiperFirmwareVersion()
time.sleep(0.5)
print("Firmware:", piper.GetPiperFirmwareVersion())
print("Before mode request:", piper.GetArmStatus())

# SDK versions expose the 0x151 mode command under different names.
mode_ctrl = getattr(piper, "ModeCtrl", None)
if mode_ctrl is None:
    mode_ctrl = piper.MotionCtrl_2

# CAN command control, MOVE J, 10% arm speed, position/velocity mode.
# No joint-position commands are sent.
for _ in range(10):
    mode_ctrl(0x01, 0x01, 10, 0x00)
    time.sleep(0.05)

print("After mode request:", piper.GetArmStatus())

if piper.GetArmStatus().arm_status.ctrl_mode != 0x01:
    raise RuntimeError(
        "Arm did not enter CAN command-control mode. "
        "Stop here and inspect the printed status."
    )

print("Current gripper feedback:")
print(piper.GetArmGripperMsgs())

for target in (20000, 30000):
    limited = piper._CalGripperSDKLimit(target, 0)
    print(f"Width target: requested={target}, after SDK limit={limited}")

original_send = piper._arm_can.SendCanMessage
seen_frames = set()

def traced_send(arbitration_id, data):
    payload = bytes(data)
    key = (arbitration_id, payload)

    if key not in seen_frames:
        seen_frames.add(key)
        print(
            f"TX id=0x{arbitration_id:03X} "
            f"len={len(payload)} data={payload.hex(' ')}"
        )

    result = original_send(arbitration_id, data)

    if result is not piper._arm_can.CAN_STATUS.SEND_MESSAGE_SUCCESS:
        print("TX failure:", result)

    return result

piper._arm_can.SendCanMessage = traced_send

for target in (20000, 30000):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        piper.GripperCtrl(
            gripper_angle=target,
            gripper_effort=500,
            gripper_code=0x01,
            set_zero=0x00,
        )
        time.sleep(0.02)

    print("Target opening (mm):", target / 1000)
    print(piper.GetArmStatus())
    print(piper.GetArmGripperMsgs())