class DeviceStateTracker:
    def __init__(self):
        self.state = {}

    def update(self, device_id: str, posture: str):
        if device_id not in self.state:
            self.state[device_id] = {
                "sit_count": 0,
                "dispensed": False,
            }

        if posture == "sitting":
            self.state[device_id]["sit_count"] += 1
        else:
            self.state[device_id]["sit_count"] = 0
            self.state[device_id]["dispensed"] = False

        return self.state[device_id]

    def mark_dispensed(self, device_id: str):
        if device_id not in self.state:
            self.state[device_id] = {"sit_count": 0, "dispensed": False}
        self.state[device_id]["dispensed"] = True