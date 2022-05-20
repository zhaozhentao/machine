package machine.entity;

import machine.enums.ParkingStatusEnum;

public class ParkingStateResult {

    public ParkingStatusEnum state;

    public ParkingStateResult(ParkingStatusEnum state) {
        this.state = state;
    }
}
