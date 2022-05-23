package machine.entity;

import machine.enums.ParkingStatusEnum;

public class ParkingStateResult {

    public ParkingStatusEnum state;

    public long timeSpent;

    public ParkingStateResult(ParkingStatusEnum state, long timeSpent) {
        this.state = state;
        this.timeSpent = timeSpent;
    }
}
