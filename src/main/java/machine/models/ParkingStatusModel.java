package machine.models;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.helper.TensorflowHelper;
import org.springframework.stereotype.Component;
import org.tensorflow.Session;

@Component
public class ParkingStatusModel {

    public Session parkingStatusSession;

    public ParkingStatusModel() throws InvalidProtocolBufferException {
        this.parkingStatusSession = TensorflowHelper.model("models/parking_status.pb");
    }
}
