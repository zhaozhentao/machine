package machine.components;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.enums.ParkingStatusEnum;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.springframework.stereotype.Component;

@Component
public class ParkingStatusModel extends BaseModel {

    public ParkingStatusModel() throws InvalidProtocolBufferException {
        super("models/parking_status.pb");
    }

    public ParkingStatusEnum predict(AutoCloseMat[] images) {
        try (
            var resizeImage = images[0];
            var ignored = images[1];
            var input = TensorflowHelper.openCVImage2Tensor(resizeImage);
            var resultTensor = s.runner().feed("input_1:0", input).fetch("output_1:0").run().get(0)
        ) {
            var result = new float[2];
            resultTensor.asRawTensor().data().asFloats().read(result);

            return result[0] > result[1] ? ParkingStatusEnum.FREE : ParkingStatusEnum.OCCUPY;
        }
    }
}
